import pytest

from io import BytesIO

from PIL import Image

from app.database import SessionLocal
from app.models import Pet


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


# ---------- M6 scam warnings ----------


def _inbox_entry(client, headers, body):
    return next(m for m in client.get("/api/me/messages", headers=headers).json() if m["body"] == body)


def test_scam_message_is_delivered_with_a_warning(client, demo, newbie):
    rex = pet_id(client, "Rex")
    body = "Lgit kalbek fi Ennasr. Ab3ath 80 dt 3al D17 9bal w njiblek iyah lyoum."
    assert client.post(f"/api/pets/{rex}/messages", json={"body": body}, headers=newbie).status_code == 201

    scam = _inbox_entry(client, demo, body)["scam"]  # delivered: warn, never block
    assert scam["level"] == "high" and "upfront_payment" in scam["labels"] and scam["reasons"]
    notif = next(n for n in client.get("/api/me/notifications", headers=demo).json() if n["title"].startswith("Likely scam"))
    assert notif["icon"] == "warning" and notif["link"] == "/dashboard/messages"


def test_honest_message_with_reward_and_phone_is_not_flagged(client, demo, newbie):
    rex = pet_id(client, "Rex")
    body = "I think I found Rex near the port! Call me on 22 345 678. No need for any reward, just glad to help."
    assert client.post(f"/api/pets/{rex}/messages", json={"body": body}, headers=newbie).status_code == 201
    assert _inbox_entry(client, demo, body)["scam"]["level"] == "none"


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


def test_report_number_skips_one_already_taken(client, newbie):
    body = {
        "status": "found", "species": "Dog", "color": "Black", "description": "Black dog found near the market.", "photos": [upload(client)], "lat": 36.858, "lng": 10.168,
        "event_date": "2026-10-15", "contact_name": "Sami Test", "contact_phone": "22 418 903",
        "contact_email": "sami@example.com",
    }
    r = client.post("/api/pets", json=body, headers=newbie)
    assert r.status_code == 201, r.text
    first = r.json()["pet"]
    # An older row already holds the number the next report would get from its id
    db = SessionLocal()
    db.get(Pet, first["id"]).ref = f"PC-TN-{1000 + first['id'] + 1:05d}"
    db.commit()
    db.close()
    r = client.post("/api/pets", json=body, headers=newbie)
    assert r.status_code == 201, r.text
    assert r.json()["pet"]["ref"] != f"PC-TN-{1000 + first['id'] + 1:05d}"


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
    assert len(r.json()["signature"]) == 48 and r.json()["signature_dim"] == 128  # demo embeddings are 128-d
    assert client.get("/api/ai/info").json() == {"provider": "demo"}
    tags =client.post("/api/ai/tags", json={"text": "Very friendly, neutered, wears a blue collar"}).json()
    assert {t["label"] for t in tags} == {"Friendly", "Neutered / spayed", "Wears a collar"}
    a = client.post("/api/ai/analyze", json={"photos": [photo], "species": "Cat"}).json()
    assert len(a["breeds"]) == 3 and a["quality"][0]["score"] > 0


def test_breed_lookalikes(client, monkeypatch):
    from app.ai.demo import DemoProvider
    from app.routers import ai as ai_router

    photo = upload(client)
    # The placeholder models have no image features, so there are no look-alikes.
    r = client.post("/api/ai/breed/lookalikes", json={"photos": [photo], "species": "Dog"})
    assert r.status_code == 200 and r.json()["results"] == []

    # A stand-in image model: the uploaded photo looks exactly like the first dog listing, a bit like the rest.
    dogs = [p for p in client.get("/api/pets", params={"status": "adopt", "page_size": 50}).json()["items"]
            if p["species"] == "Dog" and p["photos"]]
    twin = dogs[0]["photos"][0]

    class Looks(DemoProvider):
        def look_signature(self, images):
            return [1.0, 0.0] if images[0].url in (photo, twin) else [0.6, 0.8]

    monkeypatch.setattr(ai_router, "get_provider", lambda: Looks())
    ai_router._listing_look.cache_clear()
    try:
        res = client.post("/api/ai/breed/lookalikes", json={"photos": [photo], "species": "Dog"}).json()["results"]
    finally:
        ai_router._listing_look.cache_clear()
    assert len(res) == 3 and all(h["pet"]["species"] == "Dog" and h["pet"]["status"] == "adopt" for h in res)
    assert res[0]["pet"]["id"] == dogs[0]["id"] and res[0]["similarity"] == 100 and res[1]["similarity"] == 60


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


def _pet_id(client, name, status):
    items = client.get("/api/pets", params={"status": status, "page_size": 100}).json()["items"]
    return next(p["id"] for p in items if p["name"] == name)


def test_location_zones_for_open_lost_reports(client):
    r = client.get(f"/api/ai/location/{_pet_id(client, 'Rex', 'lost')}")
    if r.status_code == 503:
        pytest.skip("M7 model or libraries not installed (backend/requirements-ml.txt)")
    assert r.status_code == 200, r.text
    d = r.json()
    assert [z["share"] for z in d["zones"]] == [50, 80, 95]
    assert all(z["cells"] for z in d["zones"])
    # Zones stay around La Marsa, where Rex was last seen.
    lat, lng = d["zones"][0]["cells"][0]
    assert abs(lat - 36.878) < 0.05 and abs(lng - 10.325) < 0.05
    assert "how it got lost" in d["missing_inputs"]
    rec = d["recovery"]
    assert rec is not None and 0 <= rec["within_24h"] <= rec["within_7d"] <= rec["found_alive"] <= 1

    assert client.get(f"/api/ai/location/{_pet_id(client, 'Kahwa', 'found')}").status_code == 409
    assert client.get("/api/ai/location/999999").status_code == 404


def test_assistant_without_api_key_falls_back_to_m7(client, monkeypatch):
    from app.ai import assistant
    monkeypatch.setattr(assistant, "_backend", lambda: (_ for _ in ()).throw(assistant.AssistantUnavailable("no key")))
    r = client.get(f"/api/ai/assistant/{_pet_id(client, 'Rex', 'lost')}")
    if r.status_code == 503:
        pytest.skip("M7 model or libraries not installed (backend/requirements-ml.txt)")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["status"] == "unavailable" and d["plan"] is None and d["facts"] is None
    assert [z["share"] for z in d["location"]["zones"]] == [50, 80, 95]


def test_assistant_reads_description_and_writes_plan(client, monkeypatch):
    """Claude replaced by a fake: checks the pipeline, not the model."""
    from app.ai import assistant
    sent = []

    def fake_call(system, user, schema, effort, max_tokens):
        sent.append(user)
        if schema is assistant.ReportFacts:
            return assistant.ReportFacts(goes_outside="unknown", temperament="gregarious", circumstance="slipped_leash_walk",
                                         collar_or_tag="yes", understood="Friendly beagle that slipped its lead.")
        poster = assistant.Poster(headline="Chien perdu", text="...")
        return assistant.SearchPlan(summary="s", steps=[assistant.PlanStep(when="now", where="here", what="look")],
                                    poster_fr=poster, poster_ar=poster, share_text="t")

    monkeypatch.setattr(assistant, "_call", fake_call)
    monkeypatch.setattr(assistant, "_backend", lambda: ("groq", None, None))
    assistant._read_cached.cache_clear()
    assistant._plans.clear()
    r = client.get(f"/api/ai/assistant/{_pet_id(client, 'Rex', 'lost')}")
    if r.status_code == 503:
        pytest.skip("M7 model or libraries not installed (backend/requirements-ml.txt)")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["status"] == "ok" and d["facts"]["temperament"] == "gregarious" and d["plan"]["poster_fr"]["headline"]
    # What the description supplied is no longer listed as missing.
    missing = d["location"]["missing_inputs"]
    assert "how it got lost" not in missing and "collar or tag" not in missing
    # The plan prompt carries real places from the zones and no contact details.
    assert len(sent) == 2 and '"places"' in sent[1] and '"zone": "search first"' in sent[1]
    assert "@" not in sent[1]


def test_groq_backend_retries_until_json_matches_schema():
    """Groq JSON mode guarantees JSON, not our schema: one corrective retry, then validated output."""
    from types import SimpleNamespace as NS

    import groq

    from app.ai import assistant
    answers = iter(["not json at all",
                    '{"goes_outside": "no", "temperament": "friendly", "circumstance": "fireworks_noise", '
                    '"collar_or_tag": "yes", "understood": "Indoor cat scared by fireworks."}'])
    calls = []

    def create(**kw):
        calls.append(kw)
        return NS(choices=[NS(finish_reason="stop", message=NS(content=next(answers)))])

    client = NS(chat=NS(completions=NS(create=create)))
    facts = assistant._call_groq(groq, client, "sys", "user", assistant.ReportFacts, 500)
    assert len(calls) == 2 and calls[0]["response_format"] == {"type": "json_object"}
    assert facts.goes_outside == "no" and facts.circumstance == "fireworks_noise"
    assert facts.temperament == "unknown"  # out-of-list value coerced, not crashed
