
import os

import jwt

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from jwt.exceptions import InvalidTokenError
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User


oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/auth/login",
)

SECRET_KEY = os.environ["JWT_SECRET_KEY"]


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:

    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=["HS256"],
            options={"require": ["sub", "exp"]},
        )

        user_id = int(payload["sub"])

        if user_id <= 0:
            raise ValueError("Invalid user ID")

    except (InvalidTokenError, ValueError, TypeError):
        raise unauthorized

    user = db.get(User, user_id)

    if user is None:
        raise unauthorized

    return user
