"""Dependency-free Python client. Credentials remain in environment variables."""
import http.client
import json
import os
import time
import urllib.error
import urllib.request


class EvaluationError(RuntimeError):
    """A service failure with enough context to recover an accepted job."""

    def __init__(self, message, *, job_id=None, run_id=None, retryable=False, status=None, job=None):
        super().__init__(message)
        self.job_id = job_id
        self.run_id = run_id
        self.retryable = retryable
        self.status = status
        self.job = job

    def with_context(self, *, job_id=None, run_id=None):
        self.job_id = self.job_id or job_id
        self.run_id = self.run_id or run_id
        return self

    def __str__(self):
        context = ", ".join(f"{key}={value}" for key, value in
                            (("run", self.run_id), ("job", self.job_id)) if value)
        return super().__str__() + (f" ({context})" if context else "")


class EvaluationClient:
    def __init__(self, base_url=None, token=None, request_timeout=30):
        self.base = (base_url or os.environ.get("EVAL_URL", "http://127.0.0.1:8000")).rstrip("/")
        self.token = token or os.environ.get("EVAL_TOKEN")
        self.timeout = request_timeout
        if not self.token:
            raise EvaluationError("Set EVAL_TOKEN to the token supplied by the project owner")

    def request(self, method, path, body=None, *, timeout=None):
        data = json.dumps(body, allow_nan=False).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout if timeout is None else min(self.timeout, timeout)) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read(65536).decode(errors="replace")
            except (OSError, http.client.HTTPException):
                detail = "Unable to read HTTP error details"
            raise EvaluationError(f"HTTP {exc.code}: {detail}", status=exc.code,
                                  retryable=exc.code in (408, 429) or 500 <= exc.code < 600) from None
        except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
            raise EvaluationError(f"Request failed: {exc}", retryable=True) from None
        except (ValueError, UnicodeError) as exc:
            raise EvaluationError(f"Invalid service response: {exc}") from None

    def cases(self):
        return self.request("GET", "/v1/cases")

    def create_run(self, case_id, label, code_revision=None):
        return self.request("POST", "/v1/runs", {"case_id": case_id, "label": label, "code_revision": code_revision})

    def submit(self, run_id, products, operation="final", *, request_id=None):
        body = {"products": products, "operation": operation}
        if request_id is not None:
            body["request_id"] = request_id
        try:
            return self.request("POST", f"/v1/runs/{run_id}/jobs", body)
        except EvaluationError as exc:
            raise exc.with_context(run_id=run_id)

    def wait(self, job_id, timeout=7200, poll_seconds=1, *, run_id=None):
        deadline = time.monotonic() + timeout
        last_error = None
        while time.monotonic() < deadline:
            try:
                job = self.request("GET", f"/v1/jobs/{job_id}", timeout=max(0.001, deadline - time.monotonic()))
            except EvaluationError as exc:
                exc.with_context(job_id=job_id, run_id=run_id)
                if not exc.retryable:
                    raise
                last_error = exc
            else:
                if not isinstance(job, dict):
                    raise EvaluationError("Invalid job response returned by service", job_id=job_id, run_id=run_id)
                run_id = run_id or job.get("run_id")
                if job.get("state") == "succeeded":
                    return job
                if job.get("state") == "failed":
                    raise EvaluationError(job.get("error") or "Evaluation failed", job_id=job_id,
                                          run_id=run_id, job=job)
                if job.get("state") not in ("queued", "running"):
                    raise EvaluationError("Invalid job state returned by service", job_id=job_id, run_id=run_id)
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(max(0.01, poll_seconds), remaining))
        message = "Polling timed out; the job may still be running. Poll the same job instead of resubmitting."
        if last_error:
            message += f" Last polling error: {last_error.args[0]}"
        raise EvaluationError(message, job_id=job_id, run_id=run_id, retryable=True,
                              status=last_error.status if last_error else None)

    def evaluate(self, run_id, products, operation="final", *, request_id=None):
        job = self.submit(run_id, products, operation, request_id=request_id)
        return self.wait(job["id"], run_id=run_id)

    def run(self, run_id):
        try:
            return self.request("GET", f"/v1/runs/{run_id}")
        except EvaluationError as exc:
            raise exc.with_context(run_id=run_id)
