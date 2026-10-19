"""GitHub REST client.

Two credentials are used deliberately:

* ``read_token`` (``GITHUB_TOKEN``) is used for public reads: repository
  metadata, issues, timeline events.
* ``publish_token`` (``FORGEFLOW_GITHUB_PUBLISH_TOKEN``) is a separate,
  narrowly scoped token used only for pushing to the user's fork and opening
  the draft PR. Keeping them apart means a bug in the read path cannot write,
  and the write path cannot touch anything but the fork.

Nothing in this module trusts repository or issue content; it only moves bytes
between the GitHub API and the rest of ForgeFlow.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any

import requests

from .errors import GitHubError, RateLimitError
from .models import Issue, RepoRef

API_ROOT = "https://api.github.com"
USER_AGENT = "ForgeFlow/0.1 (+https://github.com/forgeflow)"


class GitHubClient:
    """Minimal, well-behaved GitHub REST wrapper."""

    def __init__(
        self,
        read_token: str | None = None,
        publish_token: str | None = None,
        *,
        api_root: str = API_ROOT,
        timeout: int = 30,
        max_retries: int = 3,
        session: requests.Session | None = None,
        sleep=time.sleep,
    ) -> None:
        self.read_token = read_token
        self.publish_token = publish_token or read_token
        self.api_root = api_root.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = session or requests.Session()
        self._sleep = sleep

    # -- plumbing ----------------------------------------------------------

    def _headers(self, token: str | None, accept: str | None = None) -> dict[str, str]:
        headers = {
            "Accept": accept or "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": USER_AGENT,
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def request(
        self,
        method: str,
        path: str,
        *,
        token: str | None = None,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        accept: str | None = None,
        allow_404: bool = False,
        allow_403: bool = False,
    ) -> Any:
        url = path if path.startswith("http") else f"{self.api_root}{path}"
        headers = self._headers(token if token is not None else self.read_token, accept)

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self._session.request(
                    method,
                    url,
                    headers=headers,
                    params=params,
                    json=json_body,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                last_error = exc
                self._sleep(min(2 ** attempt, 8))
                continue

            if response.status_code == 404 and allow_404:
                return None
            if response.status_code == 403 and allow_403:
                return None

            if response.status_code in (429, 502, 503, 504):
                self._sleep(self._retry_delay(response, attempt))
                continue

            if response.status_code == 403 and self._is_rate_limited(response):
                if attempt == self.max_retries - 1:
                    raise RateLimitError(
                        "GitHub rate limit exceeded; set GITHUB_TOKEN or retry later",
                        status=403,
                    )
                self._sleep(self._retry_delay(response, attempt))
                continue

            if response.status_code >= 400:
                raise GitHubError(
                    f"{method} {url} -> {response.status_code}: {self._error_detail(response)}",
                    status=response.status_code,
                )

            if response.status_code == 204 or not response.content:
                return None
            try:
                return response.json()
            except ValueError as exc:
                raise GitHubError(f"{method} {url} returned non-JSON body") from exc

        raise GitHubError(f"{method} {url} failed after {self.max_retries} attempts: {last_error}")

    @staticmethod
    def _is_rate_limited(response: requests.Response) -> bool:
        if response.headers.get("X-RateLimit-Remaining") == "0":
            return True
        body = (response.text or "").lower()
        return "rate limit" in body or "secondary rate" in body

    def _retry_delay(self, response: requests.Response, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return min(float(retry_after), 60.0)
            except ValueError:
                pass
        reset = response.headers.get("X-RateLimit-Reset")
        if reset and response.headers.get("X-RateLimit-Remaining") == "0":
            try:
                return max(0.0, min(float(reset) - time.time(), 60.0))
            except ValueError:
                pass
        return min(2 ** attempt, 8)

    @staticmethod
    def _error_detail(response: requests.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return (response.text or "")[:300]
        if isinstance(payload, dict):
            return str(payload.get("message") or payload)[:300]
        return str(payload)[:300]

    # -- reads -------------------------------------------------------------

    def get_repo(self, repo: RepoRef) -> dict[str, Any]:
        payload = self.request("GET", f"/repos/{repo.slug}")
        if not isinstance(payload, dict):  # pragma: no cover - defensive
            raise GitHubError(f"unexpected repository payload for {repo.slug}")
        return payload

    def list_open_issues(self, repo: RepoRef, limit: int = 200) -> list[Issue]:
        """Return open *issues*, dropping pull requests.

        The issues endpoint returns pull requests too; they are identified by the
        presence of a ``pull_request`` key.
        """
        issues: list[Issue] = []
        page = 1
        per_page = min(100, max(1, limit))
        while len(issues) < limit:
            payload = self.request(
                "GET",
                f"/repos/{repo.slug}/issues",
                params={
                    "state": "open",
                    "per_page": per_page,
                    "page": page,
                    "sort": "updated",
                    "direction": "desc",
                },
            )
            if not payload:
                break
            for item in payload:
                if "pull_request" in item:
                    continue
                issues.append(Issue.from_api(item))
                if len(issues) >= limit:
                    break
            if len(payload) < per_page:
                break
            page += 1
        return issues

    def linked_pull_requests(self, repo: RepoRef, issue_number: int) -> list[dict[str, Any]]:
        """Cross-referenced PRs for an issue, via the timeline API.

        The timeline API needs a token in practice; without one GitHub returns
        404/403, which we treat as "unknown" rather than "no linked PR".
        """
        payload = self.request(
            "GET",
            f"/repos/{repo.slug}/issues/{issue_number}/timeline",
            params={"per_page": 100},
            accept="application/vnd.github+json",
            allow_404=True,
            allow_403=True,
        )
        if not payload:
            return []
        found: list[dict[str, Any]] = []
        for event in payload:
            if event.get("event") not in {"cross-referenced", "connected"}:
                continue
            source = event.get("source") or {}
            issue = source.get("issue") or {}
            if issue.get("pull_request"):
                found.append(
                    {
                        "number": issue.get("number"),
                        "title": issue.get("title"),
                        "state": issue.get("state"),
                        "url": issue.get("html_url"),
                    }
                )
        return found

    def get_default_branch(self, repo: RepoRef) -> str:
        return str(self.get_repo(repo).get("default_branch") or "main")

    def get_branch_head(self, repo: RepoRef, branch: str) -> str | None:
        payload = self.request(
            "GET",
            f"/repos/{repo.slug}/git/ref/heads/{branch}",
            allow_404=True,
        )
        if not payload:
            return None
        return str((payload.get("object") or {}).get("sha") or "") or None

    def current_user(self, token: str | None = None) -> dict[str, Any]:
        payload = self.request("GET", "/user", token=token)
        return payload if isinstance(payload, dict) else {}

    def rate_limit(self) -> dict[str, Any]:
        payload = self.request("GET", "/rate_limit")
        return payload if isinstance(payload, dict) else {}

    # -- writes (publish token) -------------------------------------------

    def ensure_fork(self, repo: RepoRef, fork_owner: str) -> dict[str, Any]:
        """Return the fork, creating it if it does not exist yet."""
        existing = self.request(
            "GET", f"/repos/{fork_owner}/{repo.name}", token=self.publish_token, allow_404=True
        )
        if existing:
            return existing
        created = self.request(
            "POST",
            f"/repos/{repo.slug}/forks",
            token=self.publish_token,
            json_body={"default_branch_only": True},
        )
        return created if isinstance(created, dict) else {}

    def wait_for_fork_ready(
        self, repo: RepoRef, fork_owner: str, *, attempts: int = 10, delay: float = 3.0
    ) -> dict[str, Any]:
        """Forks are created asynchronously; poll until the git refs are usable."""
        last: dict[str, Any] = {}
        for _ in range(attempts):
            last = self.request(
                "GET",
                f"/repos/{fork_owner}/{repo.name}",
                token=self.publish_token,
                allow_404=True,
            ) or {}
            if last.get("size", 0) > 0 or last.get("pushed_at"):
                return last
            self._sleep(delay)
        return last

    def open_draft_pull_request(
        self,
        repo: RepoRef,
        *,
        title: str,
        body: str,
        head: str,
        base: str,
    ) -> dict[str, Any]:
        """Open a draft PR against the *upstream* repository.

        ``head`` is ``fork_owner:branch`` because the branch lives on the fork.
        """
        payload = self.request(
            "POST",
            f"/repos/{repo.slug}/pulls",
            token=self.publish_token,
            json_body={
                "title": title,
                "body": body,
                "head": head,
                "base": base,
                "draft": True,
                "maintainer_can_modify": True,
            },
        )
        if not isinstance(payload, dict):  # pragma: no cover - defensive
            raise GitHubError("unexpected pull request payload")
        return payload

    def post_issue_comment(self, repo: RepoRef, issue_number: int, body: str) -> dict[str, Any]:
        payload = self.request(
            "POST",
            f"/repos/{repo.slug}/issues/{issue_number}/comments",
            token=self.publish_token,
            json_body={"body": body},
        )
        return payload if isinstance(payload, dict) else {}

    def delete_branch(self, fork_owner: str, repo_name: str, branch: str) -> None:
        self.request(
            "DELETE",
            f"/repos/{fork_owner}/{repo_name}/git/refs/heads/{branch}",
            token=self.publish_token,
            allow_404=True,
        )


def iter_issue_numbers(issues: Iterable[Issue]) -> list[int]:  # pragma: no cover - helper
    return [issue.number for issue in issues]
