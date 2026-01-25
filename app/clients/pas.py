import time
import requests


class PasClient:
    def __init__(self, base_url: str, partner_id: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.partner_id = partner_id
        self.api_key = api_key
        self._access_token = None
        self._refresh_token = None
        self._access_expiry = 0

    # -----------------------
    # AUTH
    # -----------------------
    def _issue_token(self):
        url = f"{self.base_url}/api/pas/v1/auth/{self.partner_id}/token/issue"
        headers = {"X-API-Key": self.api_key, "Content-Type": "application/json"}

        resp = requests.post(url, headers=headers, timeout=10)
        resp.raise_for_status()

        data = resp.json()
        self._access_token = data["accessToken"]
        self._refresh_token = data["refreshToken"]
        self._access_expiry = time.time() + 240  # refresh 1 min before expiry

    def _refresh_access_token(self):
        url = f"{self.base_url}/api/pas/v1/auth/{self.partner_id}/token/refresh"
        headers = {"Content-Type": "application/json"}

        resp = requests.post(
            url,
            json={"refreshToken": self._refresh_token},
            headers=headers,
            timeout=10,
        )

        if resp.status_code != 200:
            self._issue_token()
            return

        data = resp.json()
        self._access_token = data["accessToken"]
        self._access_expiry = time.time() + 240

    def _ensure_token(self):
        if not self._access_token or time.time() >= self._access_expiry:
            if not self._refresh_token:
                self._issue_token()
            else:
                self._refresh_access_token()

    def _headers(self):
        self._ensure_token()
        return {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
        }

    # -----------------------
    # PUBLISH TO TOPIC
    # -----------------------
    def publish_topic_message(self, topic_name: str, payload: dict):
        url = (
            f"{self.base_url}/api/pas/v1/pubsub/{self.partner_id}"
            f"/topics/{topic_name}/messages"
        )
        resp = requests.post(url, headers=self._headers(), json={"payload": payload}, timeout=10)
        resp.raise_for_status()
        return resp.json()
