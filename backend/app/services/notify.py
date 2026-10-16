from sqlalchemy.orm import Session

from ..models import Activity, Notification


def notify(db: Session, user_id: int | None, icon: str, title: str, body: str = "", link: str = "") -> None:
    if user_id:
        db.add(Notification(user_id=user_id, icon=icon, title=title, body=body, link=link))


def log_activity(db: Session, user_id: int | None, icon: str, text: str) -> None:
    if user_id:
        db.add(Activity(user_id=user_id, icon=icon, text=text))
