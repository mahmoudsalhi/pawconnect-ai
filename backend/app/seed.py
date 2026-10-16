"""Load the demo dataset from the PawConnect design.

    python -m app.seed           # load if the database is empty
    python -m app.seed --reset   # wipe all data and reload

Demo sign-ins (password ``demo-password`` for both):
    yasmine@example.com   pet owner (Rex's report, one adoption application)
    carthage@example.com  shelter staff for Refuge Les Amis de Carthage
"""

import secrets
import sys
from datetime import date, datetime, time, timedelta, timezone

from sqlalchemy import select, text

from .database import Base, SessionLocal
from .models import (
    AccountType,
    Activity,
    AdoptionApplication,
    Favorite,
    MatchResult,
    Notification,
    Pet,
    PetPhoto,
    PetStatus,
    Shelter,
    Sighting,
    User,
)
from .security import hash_password

TUNIS = timezone(timedelta(hours=1))
TODAY = date.today()
DEMO_PASSWORD = "demo-password"
UNSPLASH = "https://images.unsplash.com/photo-"
CROPS = ["", "crop=focalpoint&fp-x=.5&fp-y=.4&fp-z=1.7", "crop=focalpoint&fp-x=.32&fp-y=.55&fp-z=1.4",
         "crop=focalpoint&fp-x=.68&fp-y=.5&fp-z=1.3"]


def photo_urls(photo_id: str) -> list[str]:
    return [UNSPLASH + photo_id + (f"?{c}" if c else "") for c in CROPS]


def at(days_back: int, hh: int, mm: int) -> datetime:
    return datetime.combine(TODAY - timedelta(days=days_back), time(hh, mm), TUNIS)


SHELTERS = [
    dict(slug="carthage", name="Refuge Les Amis de Carthage", city="Carthage", gov="Tunis", lat=36.8528, lng=10.3233,
         photo="1548199973-03cce0bbc87b", animal_count=24, since=2014, reply_note="Usually replies within a day", focus="Dogs and cats"),
    dict(slug="ariana", name="Ariana Animal Haven", city="Ariana", gov="Ariana", lat=36.8665, lng=10.1647,
         photo="1450778869180-41d0601e046e", animal_count=17, since=2018, reply_note="Usually replies within 2 days", focus="Cats and small dogs"),
    dict(slug="sousse", name="Sousse Paws Rescue", city="Sousse", gov="Sousse", lat=35.8256, lng=10.6084,
         photo="1534361960057-19889db9621e", animal_count=31, since=2016, reply_note="Usually replies within a day", focus="Senior dogs"),
    dict(slug="sfax", name="Sfax Street Cats Collective", city="Sfax", gov="Sfax", lat=34.7406, lng=10.7603,
         photo="1513245543132-31f507417b26", animal_count=42, since=2020, reply_note="Usually replies within 3 days", focus="Street cats and kittens"),
]

REPORTS = [
    dict(key="rex", name="Rex", species="Dog", breed="Beagle", age="4 years", ag="adult", sex="Male", size="Medium", color="Tricolour", status="lost", city="La Marsa", gov="Tunis", lat=36.8782, lng=10.3247, days=2, photo="1543466835-00a7907e9de1", person="Yasmine Ben Salah", role="Owner", reward="200 TND reward", chip="Yes", ref="PC-TN-00418", time="19:30", address="Corniche, La Marsa",
         desc="Rex slipped his lead near the Corniche in La Marsa on Tuesday evening around 19:30. He is friendly but nervous in crowds, answers to his name and loves cheese. Blue collar with a bone-shaped tag. Please don’t chase him. Call me and stay where you are.",
         tags=["Friendly", "Blue collar", "Microchipped", "Nervous in crowds"], ai=[["Beagle", 92], ["Basset hound", 4], ["English cocker spaniel", 2]]),
    dict(key="mishmish", name="Mishmish", species="Cat", breed="Domestic shorthair", age="3 years", ag="adult", sex="Male", size="Medium", color="Orange tabby", status="lost", city="Ennasr", gov="Ariana", lat=36.8625, lng=10.1647, days=5, photo="1573865526739-10659fec78a5", person="Ahmed Trabelsi", role="Owner", reward="100 TND reward", chip="No", ref="PC-TN-00402",
         desc="Mishmish escaped through the balcony door of our second-floor flat in Ennasr 2. He is neutered, very talkative and has a small white patch on his chest. He is an indoor cat and may be hiding under parked cars or in stairwells.",
         tags=["Indoor cat", "Neutered", "White chest patch", "Talkative"], ai=[["Abyssinian", 38], ["Bengal", 24], ["Maine Coon", 17]]),
    dict(key="max", name="Max", species="Dog", breed="Border collie mix", age="2 years", ag="adult", sex="Male", size="Large", color="Black & white", status="lost", city="Khezama", gov="Sousse", lat=35.8456, lng=10.6036, days=7, photo="1587300003388-59208cc962cb", person="Mehdi Gharbi", role="Owner", chip="Yes", ref="PC-TN-00391",
         desc="Max ran off during fireworks near Khezama beach. He is energetic, responds to whistles and wears a red harness. He was microchipped last year.",
         tags=["Red harness", "Energetic", "Microchipped", "Scared of loud noises"], ai=[["English setter", 46], ["Great Pyrenees", 19], ["Keeshond", 11]]),
    dict(key="neige", name="Neige", species="Cat", breed="Angora type", age="5 years", ag="adult", sex="Female", size="Medium", color="White", status="lost", city="Carthage", gov="Tunis", lat=36.8528, lng=10.3233, days=1, photo="1606214174585-fe31582dc6ee", person="Leila Mansour", role="Owner", chip="No", ref="PC-TN-00421",
         desc="Neige has long white fur and pale green eyes. She is shy with strangers and was last seen in the garden near Carthage Hannibal station. Spayed, no collar.",
         tags=["Shy", "Spayed", "Long white fur", "No collar"], ai=[["Persian", 44], ["Ragdoll", 29], ["Birman", 14]]),
    dict(key="bella", name="Bella", species="Dog", breed="Spaniel mix", age="3 years", ag="adult", sex="Female", size="Medium", color="Chestnut", status="lost", city="Hammamet", gov="Nabeul", lat=36.4, lng=10.6167, days=10, photo="1530281700549-e82e7bf110d6", person="Nour Ayari", role="Owner", chip="Yes", ref="PC-TN-00377",
         desc="Bella loves the beach and was last seen running along the shore near Yasmine Hammamet. Pink collar with a phone number. She is very friendly and will come to anyone with food.",
         tags=["Pink collar", "Friendly", "Microchipped", "Loves water"], ai=[["English cocker spaniel", 57], ["Wheaten terrier", 16], ["Leonberger", 9]]),
    dict(key="kahwa", name="Kahwa", species="Dog", breed="Unknown · beagle type", age="Adult (est.)", ag="adult", sex="Male", size="Medium", color="Tricolour", status="found", city="Sidi Bou Said", gov="Tunis", lat=36.8687, lng=10.3417, days=1, photo="1561037404-61cd46aa615b", person="Omar Kallel", role="Finder", chip="Not checked", ref="PC-TN-00423",
         desc="Found wandering near the Sidi Bou Said car park in the evening. No collar, very gentle and a little thin. He is staying with me until we find his family. “Kahwa” is the name my kids gave him.",
         tags=["No collar", "Gentle", "Safe with finder"], ai=[["Beagle", 71], ["Miniature pinscher", 12], ["Basset hound", 8]]),
    dict(key="ginger", name="Ginger", species="Cat", breed="Unknown · orange tabby", age="Adult (est.)", ag="adult", sex="Male", size="Medium", color="Orange tabby", status="found", city="Ennasr 2", gov="Ariana", lat=36.8545, lng=10.1702, days=3, photo="1511044568932-338cba0ad803", person="Rania Khelifi", role="Finder", chip="Not checked", ref="PC-TN-00410",
         desc="Found sleeping under a car near the Ennasr 2 market. Friendly, well fed and clearly used to people, so probably an indoor cat. Small white patch on the chest. He is safe indoors with me.",
         tags=["Friendly", "White chest patch", "Safe with finder"], ai=[["Abyssinian", 35], ["Bengal", 27], ["Maine Coon", 15]]),
    dict(key="pepper", name="Pepper", species="Dog", breed="Unknown · mixed", age="Young adult", ag="adult", sex="Male", size="Medium", color="Black & white", status="found", city="Le Bardo", gov="Tunis", lat=36.8092, lng=10.1406, days=1, photo="1477884213360-7e9d7dcc1e48", person="Karim Bouazizi", role="Finder", chip="Not checked", ref="PC-TN-00422",
         desc="Young male dog following people near the Bardo metro station. Wearing a worn black collar without a tag. Responds to whistles and seems house-trained.",
         tags=["Black collar", "House-trained", "Young"], ai=[["American pit bull terrier", 33], ["Staffordshire bull terrier", 28], ["Boxer", 12]]),
    dict(key="choco", name="Choco", species="Dog", breed="Unknown · spaniel type", age="Adult (est.)", ag="adult", sex="Female", size="Medium", color="Chestnut", status="found", city="Lac 2", gov="Tunis", lat=36.8456, lng=10.2733, days=2, photo="1544568100-847a948585b9", person="Ines Hammami", role="Finder", chip="No chip found", ref="PC-TN-00415",
         desc="Found near the lake promenade at Les Berges du Lac 2 on Tuesday morning. Calm, a little hungry, slight limp on the back left leg. A vet has checked her and she is fine.",
         tags=["Calm", "Slight limp", "Vet checked"], ai=[["English cocker spaniel", 62], ["English setter", 14], ["Leonberger", 7]]),
    dict(key="tigrou", name="Tigrou", species="Cat", breed="Unknown · tabby kitten", age="~10 weeks", ag="young", sex="Female", size="Small", color="Brown tabby", status="found", city="Sfax Médina", gov="Sfax", lat=34.7398, lng=10.76, days=6, photo="1529778873920-4da4926a72c2", person="Sfax Street Cats Collective", role="Rescue group", chip="Not checked", ref="PC-TN-00398",
         desc="Kitten found alone near Bab Diwan. Healthy, eating well and litter-trained. If no owner comes forward she will be listed for adoption.",
         tags=["Kitten", "Litter-trained", "Healthy"], ai=[["Egyptian Mau", 29], ["Bengal", 26], ["Abyssinian", 13]]),
    dict(key="safran", name="Safran", species="Cat", breed="Unknown · orange", age="Adult (est.)", ag="adult", sex="Male", size="Medium", color="Orange", status="found", city="Ariana Soghra", gov="Ariana", lat=36.8889, lng=10.1878, days=4, photo="1596854407944-bf87f6fdd49e", person="Hichem Zouari", role="Finder", chip="Not checked", ref="PC-TN-00405",
         desc="Orange cat waiting at our building door for three days. Very affectionate and has a healed scar on the left ear.",
         tags=["Affectionate", "Ear scar"], ai=[["Abyssinian", 41], ["Maine Coon", 22], ["Bengal", 18]]),
    dict(key="caramel", name="Caramel", species="Cat", breed="Unknown · tabby", age="Adult (est.)", ag="adult", sex="Female", size="Medium", color="Tabby", status="found", city="Manouba", gov="Manouba", lat=36.8101, lng=10.0956, days=8, photo="1577023311546-cdc07a8454d9", person="Mariem Sassi", role="Finder", chip="Not checked", ref="PC-TN-00386",
         desc="Tabby cat found in the university campus gardens in Manouba. Thin red collar with a bell, no tag. A little shy but lets us stroke her.",
         tags=["Red collar with bell", "Shy"], ai=[["Egyptian Mau", 37], ["Bengal", 25], ["Abyssinian", 12]]),
    dict(key="minou", name="Minou", species="Cat", breed="Unknown · tabby", age="Adult (est.)", ag="adult", sex="Female", size="Medium", color="Brown tabby", status="found", city="Lafayette", gov="Tunis", lat=36.815, lng=10.182, days=2, photo="1495360010541-f48722b34f7d", person="Sami Dridi", role="Finder", chip="Not checked", ref="PC-TN-00414",
         desc="Found in the courtyard of our building in Lafayette. Calm and curious, probably an adult female and possibly spayed.",
         tags=["Calm", "Curious"], ai=[["Egyptian Mau", 33], ["Abyssinian", 21], ["Bengal", 19]]),
]

ADOPTABLE = [
    dict(key="simba", name="Simba", species="Dog", breed="Golden retriever mix", age="1 year", ag="young", sex="Male", size="Large", color="Golden", shelter="carthage", days=15, photo="1552053831-71594a27632d", fee="50 TND", speed=1, traits=["Playful", "Good with kids", "House-trained"],
         desc="Simba was found as a puppy near Gammarth and has grown into a happy, playful young dog. He loves fetch, knows “sit” and “stay”, and gets along with children and other dogs.",
         ai=[["Leonberger", 41], ["Great Pyrenees", 22], ["Newfoundland", 13]], tips=["Listing is in good shape"]),
    dict(key="nala", name="Nala", species="Cat", breed="Persian mix", age="2 years", ag="adult", sex="Female", size="Medium", color="Calico, long hair", shelter="ariana", days=24, photo="1526336024174-e58f5cdd8e13", fee="40 TND", speed=2, traits=["Calm", "Indoor only", "Enjoys brushing"],
         desc="Nala is a gentle long-haired cat who likes quiet homes and sunny windowsills. She needs daily brushing and would suit an apartment.",
         ai=[["Persian", 63], ["Ragdoll", 18], ["Birman", 9]], tips=["Add a photo in natural light", "Mention she is litter-trained"]),
    dict(key="biscuit", name="Biscuit", species="Dog", breed="Golden mix", age="4 months", ag="young", sex="Male", size="Medium", color="Golden", shelter="ariana", days=6, photo="1507146426996-ef05306b995a", fee="50 TND", speed=1, traits=["Puppy energy", "Learning commands", "Good with dogs"],
         desc="Biscuit is a curious puppy with endless energy. He is learning basic commands and needs a family with time for walks and training.",
         ai=[["Leonberger", 36], ["Great Pyrenees", 24], ["Newfoundland", 11]], tips=["Listing is in good shape"]),
    dict(key="zitouna", name="Zitouna", species="Cat", breed="Domestic shorthair", age="3 years", ag="adult", sex="Female", size="Small", color="Silver tabby", shelter="carthage", days=48, photo="1518791841217-8f162f1e1131", fee="30 TND", speed=3, traits=["Gentle", "Lap cat", "Good with cats"],
         desc="Zitouna (“olive tree”) came to the refuge with her kittens, who have all been adopted. She is a calm lap cat who purrs the moment you sit down.",
         ai=None, tips=["Add 2 more photos", "Describe her personality in the first line"]),
    dict(key="bobby", name="Bobby", species="Dog", breed="Pug", age="7 years", ag="senior", sex="Male", size="Small", color="Black", shelter="sousse", days=96, photo="1517849845537-4d257902454a", fee="Free for seniors", speed=4, traits=["Senior", "Couch companion", "Apartment-friendly"],
         desc="Bobby is a calm senior who snores softly and loves short walks. He needs a ground-floor home or a lift because stairs are hard for him.",
         ai=[["Pug", 88], ["Boxer", 4], ["Japanese chin", 3]], tips=["Waive or lower the fee", "Add a short video", "Put “apartment-friendly” in the title"]),
    dict(key="lilou", name="Lilou", species="Cat", breed="Domestic shorthair", age="3 months", ag="young", sex="Female", size="Small", color="Tabby & white", shelter="sfax", days=4, photo="1615789591457-74a63395c990", fee="20 TND", speed=1, traits=["Curious", "Litter-trained", "Vaccinated"],
         desc="Lilou was rescued from a market stall in Sfax with her two brothers. She is curious, fearless and already uses the litter box.",
         ai=None, tips=["Listing is in good shape"]),
    dict(key="fella", name="Fella", species="Cat", breed="Domestic shorthair", age="4 years", ag="adult", sex="Female", size="Medium", color="Black & white", shelter="sousse", days=37, photo="1514888286974-6c03e2ca1dba", fee="30 TND", speed=3, traits=["Independent", "Talkative", "Spayed"],
         desc="Fella is an independent cat who likes to supervise from a high shelf. She chats when she wants dinner and prefers to be the only cat.",
         ai=None, tips=["Add more photos", "Mention she is spayed and vaccinated"]),
    dict(key="rocky", name="Rocky", species="Dog", breed="Labrador mix", age="5 years", ag="adult", sex="Male", size="Large", color="Cream", shelter="carthage", days=61, photo="1537204696486-967f1b7198c8", fee="50 TND", speed=3, traits=["Loyal", "Good on lead", "Needs a garden"],
         desc="Rocky is a loyal, easy-going dog who walks well on a lead. He would love a home with a garden and someone who enjoys long weekend walks.",
         ai=[["Great Pyrenees", 34], ["Leonberger", 21], ["American bulldog", 16]], tips=["Add photos with people", "Shorten the description to key facts"]),
    dict(key="pistache", name="Pistache", species="Dog", breed="Dachshund mix", age="2 years", ag="adult", sex="Female", size="Small", color="Red", shelter="ariana", days=19, photo="1588943211346-0908a1fb0b01", fee="40 TND", speed=2, traits=["Cuddly", "Good with kids", "Small-space friendly"],
         desc="Pistache is a cuddly little dog who follows you from room to room. She is great with children and happy in a small apartment.",
         ai=[["Miniature pinscher", 39], ["Chihuahua", 21], ["Basset hound", 14]], tips=["Add a full-body photo"]),
]

MATCHES = {
    "mishmish": {"ginger": 87, "safran": 74, "caramel": 63, "minou": 58, "tigrou": 41, "neige": 22},
    "rex": {"kahwa": 64, "choco": 47, "pepper": 39, "bella": 35, "max": 28},
    "kahwa": {"rex": 64, "bella": 33, "max": 26},
}

SIGHTINGS = {
    "rex": [dict(seen=at(0, 8, 10), place="Rue du Maroc, La Marsa", note="Beagle with a blue collar sniffing around the bakery, then headed towards the beach.", by="Anonymous"),
            dict(seen=at(1, 21, 40), place="Sidi Bou Said car park", note="Tricolour dog, ran off when I got close.", by="Omar K.")],
    "mishmish": [dict(seen=datetime(2026, 10, 13, 22, 15, tzinfo=TUNIS), place="Rue Hédi Nouira, Ennasr 2", note="Orange cat under a parked car, very vocal.", by="Rania K.")],
}


def slug_email(name: str) -> str:
    return ".".join(name.lower().replace("’", "").split()) + "@example.com"


def reset(db) -> None:
    tables = ", ".join(t.name for t in Base.metadata.sorted_tables)
    db.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))
    db.commit()


def load(db) -> None:
    shelters = {}
    for s in SHELTERS:
        sh = Shelter(slug=s["slug"], name=s["name"], city=s["city"], gov=s["gov"], lat=s["lat"], lng=s["lng"],
                     photo_url=UNSPLASH + s["photo"], animal_count=s["animal_count"], since=s["since"],
                     reply_note=s["reply_note"], focus=s["focus"])
        db.add(sh)
        shelters[s["slug"]] = sh
    db.flush()

    demo = User(name="Yasmine Ben Salah", email="yasmine@example.com", password_hash=hash_password(DEMO_PASSWORD),
                account_type=AccountType.owner, phone="+216 22 418 903", gov="Tunis",
                created_at=datetime(2025, 3, 2, tzinfo=TUNIS))
    staff = User(name="Amira Haddad", email="carthage@example.com", password_hash=hash_password(DEMO_PASSWORD),
                 account_type=AccountType.shelter, gov="Tunis", shelter_id=shelters["carthage"].id)
    db.add_all([demo, staff])
    # Everyone else who published a report gets an account that can't sign in (random password).
    people: dict[str, User] = {demo.name: demo}
    for r in REPORTS:
        if r["person"] not in people and r["role"] != "Rescue group":
            u = User(name=r["person"], email=slug_email(r["person"]), password_hash=hash_password(secrets.token_urlsafe(24)),
                     gov=r["gov"], created_at=datetime(2025, 6, 1, tzinfo=TUNIS))
            db.add(u)
            people[r["person"]] = u
    db.flush()

    pets: dict[str, Pet] = {}
    for r in REPORTS:
        owner = people.get(r["person"])
        p = Pet(ref=r["ref"], status=PetStatus(r["status"]), name=r["name"], species=r["species"], breed=r["breed"],
                age_label=r["age"], age_group=r["ag"], sex=r["sex"], size=r["size"], color=r["color"], chip=r["chip"],
                description=r["desc"], tags=r["tags"], reward=r.get("reward"), ai_breeds=r["ai"], city=r["city"],
                gov=r["gov"], lat=r["lat"], lng=r["lng"], address=r.get("address", ""), event_date=TODAY - timedelta(days=r["days"]),
                event_time=r.get("time", ""), contact_name=r["person"], contact_role=r["role"],
                contact_email=owner.email if owner else "", owner_id=owner.id if owner else None,
                shelter_id=shelters["sfax"].id if r["role"] == "Rescue group" else None,
                views=342 if r["key"] == "rex" else 40 + r["days"] * 23, created_at=at(r["days"], 20, 0))
        p.photos = [PetPhoto(url=u, position=i) for i, u in enumerate(photo_urls(r["photo"]))]
        db.add(p)
        pets[r["key"]] = p

    for a in ADOPTABLE:
        sh = shelters[a["shelter"]]
        default_ai = [["Egyptian Mau", 34], ["Bengal", 22], ["Abyssinian", 14]] if a["species"] == "Cat" else \
            [["American pit bull terrier", 31], ["Staffordshire bull terrier", 24], ["Boxer", 11]]
        p = Pet(ref=f"PC-AD-0{1180 + a['days']}", status=PetStatus.adopt, name=a["name"], species=a["species"],
                breed=a["breed"], age_label=a["age"], age_group=a["ag"], sex=a["sex"], size=a["size"], color=a["color"],
                chip="Yes", description=a["desc"], tags=a["traits"], traits=a["traits"], fee=a["fee"],
                ai_breeds=a["ai"] or default_ai, adopt_speed=a["speed"], listing_tips=a["tips"], city=sh.city, gov=sh.gov,
                lat=sh.lat, lng=sh.lng, event_date=TODAY - timedelta(days=a["days"]), contact_name=sh.name,
                contact_role="Shelter", contact_email=f"{sh.slug}@example.com", shelter_id=sh.id,
                owner_id=staff.id if sh.slug == "carthage" else None, views=60 + a["days"] * 4,
                created_at=at(a["days"], 10, 0))
        p.photos = [PetPhoto(url=u, position=i) for i, u in enumerate(photo_urls(a["photo"]))]
        db.add(p)
        pets[a["key"]] = p

    reunited = Pet(ref="PC-TN-00329", status=PetStatus.found, resolved=True, resolved_at=datetime(2026, 9, 4, 18, 0, tzinfo=TUNIS),
                   name="Found tabby cat", species="Cat", breed="Unknown · tabby", age_label="Adult (est.)", sex="Unknown",
                   color="Tabby", description="Tabby cat found sheltering by the café steps in Sidi Bou Said. Calm and well fed.",
                   city="Sidi Bou Said", gov="Tunis", lat=36.8708, lng=10.3472, event_date=date(2026, 9, 2),
                   contact_name=demo.name, contact_role="Finder", owner_id=demo.id, views=118,
                   created_at=datetime(2026, 9, 2, 12, 0, tzinfo=TUNIS))
    reunited.photos = [PetPhoto(url=UNSPLASH + "1592194996308-7b43878e84a6", position=0)]
    db.add(reunited)
    db.flush()

    for q, cands in MATCHES.items():
        for c, score in cands.items():
            db.add(MatchResult(query_pet_id=pets[q].id, candidate_pet_id=pets[c].id, score=score,
                               seen=not (q == "rex" and c == "kahwa"), created_at=at(0 if c == "kahwa" else 1, 7, 52)))
    for key, rows in SIGHTINGS.items():
        for s in rows:
            db.add(Sighting(pet_id=pets[key].id, place=s["place"], seen_at=s["seen"], note=s["note"], by_name=s["by"], created_at=s["seen"]))

    db.add_all([Favorite(user_id=demo.id, pet_id=pets["simba"].id), Favorite(user_id=demo.id, pet_id=pets["nala"].id)])
    db.add(AdoptionApplication(code="APP-2041", pet_id=pets["nala"].id, user_id=demo.id, full_name=demo.name,
                               phone=demo.phone, gov="Tunis", housing="Apartment", other_pets="None", hours_alone="4–6 hours",
                               motivation="I work from home most days and have shared my flat with a long-haired cat before.",
                               home_visit_ok=True, stage=2, next_step="Home visit proposed for Sunday 19 Oct at 10:00.",
                               created_at=datetime(2026, 10, 8, 11, 0, tzinfo=TUNIS)))

    rex, kahwa, nala, simba = pets["rex"], pets["kahwa"], pets["nala"], pets["simba"]
    now = datetime.now(TUNIS)
    notes = [
        ("auto_awesome", "New possible match for Rex", "A dog found in Sidi Bou Said scored 64% visual similarity.",
         f"/match?pet={rex.id}&selected={kahwa.id}", True, now - timedelta(minutes=12)),
        ("visibility", "New sighting of Rex", "Rue du Maroc, La Marsa · today at 08:10", f"/pets/{rex.id}", True, now - timedelta(hours=2)),
        ("event_available", "Home visit scheduled for Nala", "Ariana Animal Haven proposed Sunday 19 Oct at 10:00.",
         "/dashboard/applications", True, now - timedelta(days=1)),
        ("chat", "Message from Omar K.", "“The dog I found has a white tip on his tail. Does Rex?”", f"/pets/{rex.id}", False,
         now - timedelta(days=1, hours=3)),
        ("favorite", "Simba has new photos", "A pet you saved was updated by Refuge Les Amis de Carthage.", f"/pets/{simba.id}",
         False, now - timedelta(days=2)),
    ]
    for icon, title, body, link, unread, ts in notes:
        db.add(Notification(user_id=demo.id, icon=icon, title=title, body=body, link=link, unread=unread, created_at=ts))
    for icon, txt, ts in [
        ("visibility", "Someone reported a sighting of Rex in La Marsa", at(0, 8, 10)),
        ("auto_awesome", "AI matching found a new possible match for Rex", at(0, 7, 52)),
        ("event_available", "Ariana Animal Haven proposed a home visit", at(1, 16, 0)),
        ("favorite", "You saved Simba", at(3, 12, 0)),
        ("campaign", "You published a lost report for Rex", at(2, 20, 5)),
    ]:
        db.add(Activity(user_id=demo.id, icon=icon, text=txt, created_at=ts))
    db.commit()
    print(f"Seeded {len(pets) + 1} pets, {len(shelters)} shelters, {len(people) + 1} users.")
    print(f"Demo sign-in: yasmine@example.com / {DEMO_PASSWORD}")


def main() -> None:
    db = SessionLocal()
    try:
        if "--reset" in sys.argv:
            reset(db)
        elif db.scalar(select(Shelter).limit(1)):
            print("Database already has data. Use --reset to reload the demo dataset.")
            return
        load(db)
    finally:
        db.close()


if __name__ == "__main__":
    main()
