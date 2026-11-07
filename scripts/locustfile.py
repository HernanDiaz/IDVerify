"""Locust load test for the DocVerify API.

Run against a started stack:
    locust -f scripts/locustfile.py --host http://localhost:8000
"""

from __future__ import annotations

import io

from locust import HttpUser, between, task
from PIL import Image


def _image_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (512, 320), (180, 170, 160)).save(buf, format="JPEG")
    return buf.getvalue()


_IMAGE = _image_bytes()


class DocVerifyUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task(3)
    def submit_and_poll(self) -> None:
        resp = self.client.post(
            "/analyses", files={"file": ("doc.jpg", _IMAGE, "image/jpeg")}
        )
        if resp.status_code == 201:
            analysis_id = resp.json()["id"]
            self.client.get(f"/analyses/{analysis_id}", name="/analyses/{id}")

    @task(1)
    def list_analyses(self) -> None:
        self.client.get("/analyses?limit=10", name="/analyses")

    @task(1)
    def health(self) -> None:
        self.client.get("/health")
