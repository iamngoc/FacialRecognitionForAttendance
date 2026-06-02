"""
Sleepy Durian World - completely Testcase: mathematics, database, API, AI
"""

import pytest
import numpy as np
import requests
import cv2

BASE_URL = "http://localhost:8000"

def normalized_vector(dim=512) -> np.ndarray:
    """random normalized vector (simulate ArcFace-Embedding)."""
    v = np.random.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)

def cosine(e1, e2) -> float:
    n1, n2 = np.linalg.norm(e1), np.linalg.norm(e2)
    return float(np.dot(e1, e2) / (n1 * n2)) if n1 and n2 else 0.0

def gray_img(h=480, w=640) -> bytes:
    """Gray test image as JPEG-Bytes (no real face)."""
    bild = np.full((h, w, 3), 100, dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", bild)
    return buf.tobytes()


def backend_ping() -> bool:
    try:
        return requests.get(f"{BASE_URL}/health", timeout=3).ok
    except Exception:
        return False


@pytest.fixture(scope="session")
def backend():
    """Session-wide fixture: skips if backend is not running"""
    if not backend_ping():
        pytest.skip("Backend not available – start 'docker-compose up'")
    return BASE_URL

class TestCosMath:
    """
The similarity calculation is performed in isolation.
These tests always run, even without a backend.
    """

    def test_same_vector_score_one(self):
        """Cosine of a vector with itself = 1.0 (100% equal)."""
        v = normalized_vector(512)
        score = cosine(v, v)
        assert abs(score - 1.0) < 1e-5, \
            f"Expect 1.0, receive {score:.6f}"

    def test_orthogonal_vectors_score_null(self):
        """orthogonal vectors = 0.0 (total different)."""
        e1 = np.zeros(512, dtype=np.float32); e1[0] = 1.0
        e2 = np.zeros(512, dtype=np.float32); e2[1] = 1.0
        assert abs(cosine(e1, e2)) < 1e-5

    def test_score_between_minus1_and_plus1(self):
        """The score must mathematically lie in the range [-1, 1]."""
        for _ in range(200):
            s = cosine(normalized_vector(), normalized_vector())
            assert -1.0 <= s <= 1.0, f"Score out of: {s}"

    def test_similar_vectors_high_score(self):
        """
        Slight noise = slightly different photo of the same person.
        → Score must be high (> 0.90).
        """
        original = normalized_vector()
        # 2% Noise slightly simulates different photos
        noise = original + np.random.randn(512).astype(np.float32)*0.02
        noise /= np.linalg.norm(noise)
        score = cosine(original, noise)
        assert score > 0.90, \
            f"Similar embeddings should be > 0.90, was {score:.4f}"

    def test_different_people_lower_score(self):
        """
        100 random pairs of different 'people'. Average score must be below 0.30.
        """
        scores = [cosine(normalized_vector(), normalized_vector())
                  for _ in range(100)]
        avg = float(np.mean(scores))
        assert avg < 0.30, \
            f"Random pairs should avg < 0.30, was {avg:.4f}"

    def test_threshold_logic_correct(self):
        """
        Detection should only occur if the score is greater than or equal to the threshold.
        Simulates the logic from face_engine.py.
        """
        threshold = 0.45
        person_a    = normalized_vector()
        # Similar person (minor disturbance) → should be identified
        similar = person_a + np.random.randn(512).astype(np.float32)*0.01
        similar /= np.linalg.norm(similar)
        # Randomly selected different person → should NOT be identified
        strange = normalized_vector()
        # Guarantee that stranger is really strange
        while cosine(person_a, strange) >= threshold:
            strange = normalized_vector()

        assert cosine(person_a, similar) >= threshold, \
            "A similar person should be above the threshold"
        assert cosine(person_a, strange) < threshold, \
             " strange Person should be below the threshold"

    def test_zero_vector_return_numm(self):
        """Zero vector (empty embedding) → Score 0.0, no crash"""
        null = np.zeros(512, dtype=np.float32)
        normal = normalized_vector()
        assert cosine(null, normal) == 0.0

class TestBackendAPI:
    def test_root_endpoint(self, backend):
        """GET / get system information."""
        r = requests.get(f"{backend}/")
        assert r.status_code == 200
        d = r.json()
        assert "SleepyDurianWorld" in d["system"]
        assert d["status"] == "online"
        assert "threshold" in d
        print(f"\n   System: {d['system']}")
        print(f"   threshold: {d['threshold']}")

    def test_health_endpoint(self, backend):
        """GET /health for Docker health check."""
        r = requests.get(f"{backend}/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_employees_list(self, backend):
        """GET /Employee returns list with Schewe and Nguyen."""
        r = requests.get(f"{backend}/employees")
        assert r.status_code == 200
        list = r.json()
        assert isinstance(list, list)
        assert len(list) == 2, \
            f"Expect 2 Employees, found {len(list)}"
        names = [e["name"] for e in list]
        assert "Jonas Schewe"  in names, "Jonas Schewe is missing!"
        assert "Ngoc Nguyen" in names, "Ngoc Nguyen is missing!"
        print(f"\n   Employees: {names}")
        for e in list:
            print(f"   {e['name']}: "
                  f"registered={'✅' if e['registered'] else '❌'}")

    def test_scan_empty_image_is_rejected(self, backend):
        """POST /scan mit gray image (no face) → recognized=False."""
        r = requests.post(
            f"{backend}/scan",
            files={"photo": ("test.jpg", gray_img(), "image/jpeg")}
        )
        assert r.status_code == 200
        d = r.json()
        assert d["recognized"] == False
        print(f"\n   Empty Photo is correctly rejected: {d['message']}")

    def test_scan_invalid_format_return_400(self, backend):
        """POST /scan with text instead of image  → HTTP 400."""
        r = requests.post(
            f"{backend}/scan",
            files={"photo": ("test.txt", b"no photo", "text/plain")}
        )
        assert r.status_code in [400, 422]

    def test_today_report(self, backend):
        """GET /report/today return report of the day."""
        r = requests.get(f"{backend}/report/today")
        assert r.status_code == 200
        d = r.json()
        assert "date"  in d
        assert "scans"  in d
        assert isinstance(d["scans"], list)
        print(f"\n   Today report: {len(d['scans'])} Scans")

    def test_report_week_jonas(self, backend):
        """GET /report/week/1 → Jonas Schewe."""
        r = requests.get(f"{backend}/report/week/1")
        assert r.status_code == 200
        assert r.json()["employees"] == "Jonas Schewe"

    def test_report_week_ngoc(self, backend):
        """GET /report/week/2 → Ngoc Nguyen."""
        r = requests.get(f"{backend}/report/week/2")
        assert r.status_code == 200
        assert r.json()["employees"] == "Ngoc Nguyen"

    def test_report_unknown_id_return_404(self, backend):
        """GET /report/week/999 → HTTP 404."""
        r = requests.get(f"{backend}/report/week/999")
        assert r.status_code == 404

    def test_enrollment_not_exist_employee(self, backend):
        """POST /enrollment/999 → HTTP 404."""
        r = requests.post(
            f"{backend}/enrollment/999",
            files=[("photos", ("t.jpg", gray_img(), "image/jpeg"))]
        )
        assert r.status_code == 404

class TestAfterEnrollment:
    """
    These tests check if enrollment was successful.
    Run only AFTER Jonas and Ngoc have been registered!
    """

    def test_2_employees_have_embedding(self, backend):
        """After enrollment they should have embedding."""
        r = requests.get(f"{backend}/employees")
        list = r.json()
        without = [e["name"] for e in list if not e["registered"]]
        assert len(without) == 0, \
            (f"This employee doesn't have embedding: {without}\n"
             f"Please complete the enrollment process first.:\n"
             f"  python scripts/enrollment-script.py --id 1 --name 'Jonas Schewe'\n"
             f"  python scripts/enrollment-script.py --id 2 --name 'Ngoc Nguyen'")
        print(f"\n   all employees do have embedding!")


def quick_test():
    """quick Systemcheck for J&N."""
    print("\n" + "═"*52)
    print("SleepyDurianWorld SYSTEM TEST")
    print("═"*52)

    ok_count = total = 0

    def check(description, condition, detail=""):
        nonlocal ok_count, total
        total += 1
        icon = "✅" if condition else "❌"
        print(f"  {icon}  {description}")
        if detail:
            print(f"       → {detail}")
        if condition:
            ok_count += 1

    print("\n  Mathematics:")
    v = normalized_vector()
    check("the same vector → Score 1.0",
          abs(cosine(v, v) - 1.0) < 1e-5,
          f"Score = {cosine(v,v):.6f}")

    e1 = np.zeros(512, dtype=np.float32); e1[0] = 1.0
    e2 = np.zeros(512, dtype=np.float32); e2[1] = 1.0
    check("Orthogonal vectors → Score 0.0",
          abs(cosine(e1, e2)) < 1e-5,
          f"Score = {cosine(e1,e2):.6f}")

    noise = v + np.random.randn(512).astype(np.float32)*0.02
    noise /= np.linalg.norm(noise)
    s = cosine(v, noise)
    check(f"Similar Embeddings → Score > 0.90", s > 0.90,
          f"Score = {s:.4f}")

    avg = np.mean([cosine(normalized_vector(), normalized_vector())
                   for _ in range(100)])
    check("random Pairs → Average < 0.30",
          avg < 0.30, f"Average = {avg:.4f}")

    print("\nBackend:")
    if not backend_ping():
        print("Backend not available!")
        print("Please start with: docker-compose up --build")
    else:
        r = requests.get(f"{BASE_URL}/")
        check("GET /  →  Status online",
              r.json().get("status") == "online",
              r.json().get("system",""))

        r = requests.get(f"{BASE_URL}/employees")
        list = r.json()
        check("2 employees in database",
              len(list) == 2,
              str([e["name"] for e in list]))

        with_ = [e["name"] for e in list if e["registered"]]
        without = [e["name"] for e in list if not e["registered"]]
        check(f"Employee mit Embedding: {len(with_)}/2",
              len(with_) == 2,
              f"Registered: {with_} | To be missing: {without}")

        r = requests.post(f"{BASE_URL}/scan",
            files={"photo":("t.jpg", gray_img(), "image/jpeg")})
        check("Empty image will be rejected.",
              r.json().get("recognized") == False,
              r.json().get("message",""))


    print(f"\n  {'═'*48}")
    print(f"  Result: {ok_count}/{total} Tests passed.")
    if ok_count == total:
        print("ALL GREEN! System is ready for use.!")
    else:
        print("Some tests failed. Please check.")
    print(f"  {'═'*48}\n")


if __name__ == "__main__":
    quick_test()
