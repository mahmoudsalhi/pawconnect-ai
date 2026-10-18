from io import BytesIO

from PIL import Image


def pet_id(client, name):
    items = client.get("/api/pets", params={"q": name, "page_size": 50}).json()["items"]
    return next(p["id"] for p in items if p["name"] == name)


def upload(client):
    buf = BytesIO()
    Image.new("RGB", (320, 240), (180, 120, 70)).save(buf, "JPEG")
    r = client.post("/api/uploads", files=[("files", ("cat.jpg", buf.getvalue(), "image/jpeg"))])
    assert r.status_code == 201, r.text
    return r.json()["urls"][0]


# ---------- auth ----------


def test_register_login_and_duplicate(client):
    body = {"name": "Leila Test", "email": "leila.test@example.com", "password": "longpassword"}
    assert client.post("/api/auth/register", json=body).status_code == 201
    r = client.post("/api/auth/register", json=body)
    assert r.status_code == 409
    r = client.post("/api/auth/login", json={"email": "LEILA.TEST@example.com", "password": "longpassword"})
    assert r.status_code == 200
    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {r.json()['access_token']}"}).json()
    assert me["name"] == "Leila Test"


def test_login_rejects_bad_password(client):
    r = client.post("/api/auth/login", json={"email": "yasmine@example.com", "password": "nope-nope"})
    assert r.status_code == 401


def test_register_validation_messages(client):
    r = client.post("/api/auth/register", json={"name": "A", "email": "not-an-email", "password": "short"})
    assert r.status_code == 422
    errs = r.json()["errors"]
    assert set(errs) >= {"name", "email", "password"}
    assert errs["password"] == "Use at least 8 characters."


def test_protected_routes_need_a_token(client):
    assert client.get("/api/me/dashboard").status_code == 401


# ---------- browse ----------


def test_browse_filters_and_counts(client):
    d = client.get("/api/pets").json()
    assert d["counts"] == {"all": 28, "lost": 5, "found": 8, "adopt": 15}
    assert d["total"] == 28 and len(d["items"]) == 8 and d["pages"] == 4

    cats = client.get("/api/pets", params={"species": "Cat", "status": "found"}).json()
    assert cats["total"] == 5 and all(p["species"] == "Cat" for p in cats["items"])

    near = client.get("/api/pets", params={"distance": 5, "sort": "distance", "page_size": 50}).json()["items"]
    assert near and all(p["distance_km"] <= 5 for p in near)
    assert [p["distance_km"] for p in near] == sorted(p["distance_km"] for p in near)

    assert client.get("/api/pets", params={"q": "good with kids", "status": "adopt"}).json()["total"] == 4


def test_resolved_reports_are_hidden(client):
    names = [p["name"] for p in client.get("/api/pets", params={"page_size": 100}).json()["items"]]
    assert "Found tabby cat" not in names


def test_detail_hides_phone_and_counts_views(client):
    rex = pet_id(client, "Rex")
    a = client.get(f"/api/pets/{rex}").json()
    b = client.get(f"/api/pets/{rex}").json()
    assert b["views"] == a["views"] + 1
    assert a["contact_phone"] is None
    assert len(a["sightings"]) == 2 and a["contact_note"].startswith("Phone verified")
    assert len(client.get(f"/api/pets/{rex}/similar").json()) == 4
    assert client.get("/api/pets/999999").status_code == 404


# ---------- favourites, sightings, messages ----------


def test_favorites_roundtrip(client, newbie):
    rex = pet_id(client, "Rex")
    assert client.put(f"/api/me/favorites/{rex}", headers=newbie).status_code == 204
    assert client.put(f"/api/me/favorites/{rex}", headers=newbie).status_code == 204  # idempotent
    favs = client.get("/api/me/favorites", headers=newbie).json()
    assert [p["name"] for p in favs] == ["Rex"] and favs[0]["is_favorite"]
    client.delete(f"/api/me/favorites/{rex}", headers=newbie)
    assert client.get("/api/me/favorites", headers=newbie).json() == []


def test_sighting_notifies_owner(client, demo):
    rex = pet_id(client, "Rex")
    before = len(client.get("/api/me/notifications", headers=demo).json())
    r = client.post(f"/api/pets/{rex}/sightings", json={"place": "Avenue Habib Bourguiba", "note": "Heading north"})
    assert r.status_code == 201 and r.json()["by_name"] == "Anonymous"
    notifs = client.get("/api/me/notifications", headers=demo).json()
    assert len(notifs) == before + 1 and notifs[0]["title"] == "New sighting of Rex"
    assert client.post(f"/api/pets/{pet_id(client, 'Kahwa')}/sightings", json={"place": "Somewhere"}).status_code == 400


def test_message_reaches_owner(client, demo, newbie):
    rex = pet_id(client, "Rex")
    r = client.post(f"/api/pets/{rex}/messages", json={"body": "I think I saw him near the port."}, headers=newbie)
    assert r.status_code == 201
    inbox = client.get("/api/me/messages", headers=demo).json()
    assert inbox[0]["body"] == "I think I saw him near the port." and inbox[0]["pet_name"] == "Rex"
    assert client.post(f"/api/pets/{rex}/messages", json={"body": "hi"}, headers=newbie).status_code == 422


# ---------- reports & matching ----------


def test_publish_found_report_matches_and_notifies(client, newbie):
    photo = upload(client)
    body = {
        "status": "found", "species": "Cat", "color": "Orange tabby", "size": "Medium",
        "description": "Friendly orange tabby with a white chest patch, found near the market.",
        "photos": [photo], "lat": 36.858, "lng": 10.168, "address": "Rue Hédi Nouira, Ennasr",
        "event_date": "2026-10-15", "contact_name": "Sami Test", "contact_phone": "22 418 903",
        "contact_email": "sami@example.com", "tags": ["Friendly"],
    }
    r = client.post("/api/pets", json=body, headers=newbie)
    assert r.status_code == 201, r.text
    out = r.json()
    assert out["pet"]["ref"].startswith("PC-TN-") and out["pet"]["gov"] == "Ariana"
    assert out["pet"]["contact_phone"] == "+216 22 418 903"  # visible to its owner
    assert out["pool"] == 2  # open lost cats

    dash = client.get("/api/me/dashboard", headers=newbie).json()
    assert dash["reports"][0]["pet"]["id"] == out["pet"]["id"]

    bad = client.post("/api/pets", json=body | {"contact_phone": "123"}, headers=newbie)
    assert bad.status_code == 422 and "8-digit" in bad.json()["errors"]["contact_phone"]


def test_lost_report_requires_name(client, newbie):
    body = {
        "status": "lost", "species": "Dog", "color": "Black", "description": "Black dog with a red collar, very shy.",
        "photos": ["/uploads/x.jpg"], "lat": 36.8, "lng": 10.2, "event_date": "2026-10-15",
        "contact_name": "Sami Test", "contact_phone": "22418903", "contact_email": "sami@example.com",
    }
    assert client.post("/api/pets", json=body, headers=newbie).status_code == 422


def test_owner_can_resolve_but_others_cannot(client, demo, newbie):
    rex = pet_id(client, "Rex")
    assert client.post(f"/api/pets/{rex}/resolve", headers=newbie).status_code == 403


# ---------- adoption ----------


def test_adoption_application_flow(client, newbie, staff):
    simba = pet_id(client, "Simba")
    body = {"pet_id": simba, "full_name": "Sami Test", "phone": "+216 22 418 903", "gov": "Tunis", "housing": "Apartment",
            "other_pets": "None", "hours_alone": "2–4 hours", "home_visit_ok": True,
            "motivation": "I work from home and have looked after dogs for years."}
    assert client.post("/api/adoptions", json=body | {"home_visit_ok": False}, headers=newbie).status_code == 422
    r = client.post("/api/adoptions", json=body, headers=newbie)
    assert r.status_code == 201, r.text
    app = r.json()
    assert app["code"].startswith("APP-") and app["stage_label"] == "Submitted"
    assert client.post("/api/adoptions", json=body, headers=newbie).status_code == 409

    incoming = client.get("/api/adoptions/incoming", headers=staff).json()
    assert any(a["id"] == app["id"] for a in incoming)
    assert client.patch(f"/api/adoptions/{app['id']}", json={"stage": 1}, headers=newbie).status_code == 403
    r = client.patch(f"/api/adoptions/{app['id']}", json={"stage": 2, "next_step": "Home visit on Saturday."}, headers=staff)
    assert r.json()["stage_label"] == "Home visit"
    mine = client.get("/api/adoptions/mine", headers=newbie).json()
    assert mine[0]["next_step"] == "Home visit on Saturday."


def test_shelters_and_insights(client):
    shelters = client.get("/api/shelters").json()
    assert [s["slug"] for s in shelters] == ["carthage", "ariana", "sousse", "sfax"]
    rows = client.get("/api/shelters/carthage/insights").json()
    speeds = [r["speed_class"] for r in rows]
    assert speeds == sorted(speeds, reverse=True) and all(0 <= s <= 4 for s in speeds)
    assert all(r["tips"] for r in rows)


# ---------- AI ----------


def test_ai_match_uses_cached_scores(client):
    mishmish = pet_id(client, "Mishmish")
    r = client.post("/api/ai/match", json={"pet_id": mishmish, "species": "Cat", "status": "found", "area": "ennasr",
                                           "radius": 20, "threshold": 50}).json()
    seeded = [(h["pet"]["name"], h["score"]) for h in r["results"] if h["pet"]["ref"] < "PC-TN-01000"]
    assert seeded == [("Ginger", 87), ("Safran", 74), ("Caramel", 63), ("Minou", 58)]
    assert len(r["signature"]) == 48


def test_ai_compatibility(client):
    r = client.post("/api/ai/compatibility", json={"species": "Cat", "home": "apartment", "hours_alone": "over8"}).json()
    scores = [h["score"] for h in r["results"]]
    assert r["results"] and all(h["pet"]["species"] == "Cat" for h in r["results"])
    assert scores == sorted(scores, reverse=True) and all(0 <= s <= 100 for s in scores)
    assert all(isinstance(x["good"], bool) and x["text"] for h in r["results"] for x in h["reasons"])

    family = client.post("/api/ai/compatibility", json={"children": "young", "activity": "relaxed"}).json()["results"]
    runner = client.post("/api/ai/compatibility", json={"activity": "active", "home": "house_garden"}).json()["results"]
    assert family[0]["pet"]["name"] != runner[0]["pet"]["name"]

    bad = client.post("/api/ai/compatibility", json={"home": "castle"})
    assert bad.status_code == 422


def test_ai_match_by_photo_and_tags(client):
    photo = upload(client)
    r = client.post("/api/ai/match", json={"photo": photo, "species": "Dog", "status": "both", "area": "marsa",
                                           "radius": 300, "threshold": 0})
    assert r.status_code == 200 and all(h["pet"]["species"] == "Dog" for h in r.json()["results"])
    tags = client.post("/api/ai/tags", json={"text": "Very friendly, neutered, wears a blue collar"}).json()
    assert {t["label"] for t in tags} == {"Friendly", "Neutered / spayed", "Wears a collar"}
    a = client.post("/api/ai/analyze", json={"photos": [photo], "species": "Cat"}).json()
    assert len(a["breeds"]) == 3 and a["quality"][0]["score"] > 0


def test_uploaded_photo_is_served_from_the_database(client):
    url = upload(client)
    r = client.get(url)
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert Image.open(BytesIO(r.content)).size == (320, 240)
    assert client.get("/uploads/missing.jpg").status_code == 404


def test_upload_rejects_non_images(client):
    r = client.post("/api/uploads", files=[("files", ("x.jpg", b"not an image", "image/jpeg"))])
    assert r.status_code == 422


def test_settings_update(client, newbie):
    r = client.patch("/api/me", json={"phone": "22418903", "min_match_score": 75, "lang": "Français"}, headers=newbie)
    assert r.status_code == 200 and r.json()["phone"] == "+216 22 418 903" and r.json()["min_match_score"] == 75
    assert client.patch("/api/me", json={"email": "yasmine@example.com"}, headers=newbie).status_code == 409


def test_owner_can_dismiss_match(client, demo, newbie):
    rex, choco = pet_id(client, "Rex"), pet_id(client, "Choco")
    assert client.post(f"/api/ai/match/{rex}/dismiss/{choco}", headers=newbie).status_code == 403
    assert client.post(f"/api/ai/match/{rex}/dismiss/{choco}", headers=demo).status_code == 204
    r = client.post("/api/ai/match", json={"pet_id": rex, "species": "Dog", "status": "found", "area": "marsa",
                                           "radius": 300, "threshold": 0}, headers=demo).json()
    assert "Choco" not in [h["pet"]["name"] for h in r["results"]]
    dash = client.get("/api/me/dashboard", headers=demo).json()
    assert all(not m["is_new"] for m in dash["matches"])  # running the match marks results as seen
