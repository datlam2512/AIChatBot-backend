
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
import jwt
from jwt.exceptions import InvalidTokenError
from app.services.security import SECRET_KEY, ALGORITHM
from app.database import get_db
from app.models import User
from app.schemas.auth import RegisterRequest, TokenResponse, UserResponse,LoginRequest
from app.services.security import (
    create_access_token,
    hash_password,
    verify_password,
)
from fastapi.security import (
    HTTPBearer,
    HTTPAuthorizationCredentials,
)
bearer_scheme = HTTPBearer(
    scheme_name="JWT Authentication",
    bearerFormat="JWT",
)



def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(
        bearer_scheme
    ),
    db: Session = Depends(get_db),
) -> User:

    token = credentials.credentials

    credentials_exception = HTTPException(
        status_code=401,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM],
            options={"require": ["sub", "exp"]},
        )

        user_id = int(payload["sub"])

        if user_id <= 0:
            raise ValueError("Invalid user ID")

    except (InvalidTokenError, ValueError, TypeError):
        raise credentials_exception

    user = db.get(User, user_id)

    if user is None:
        raise credentials_exception

    return user




router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)


# =====================================
# REGISTER
# =====================================

@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
def register(
    request: RegisterRequest,
    db: Session = Depends(get_db),
):
    email = str(request.email).strip().lower()

    # Kiểm tra email đã tồn tại chưa
    existing_user = db.scalar(
        select(User).where(User.email == email)
    )

    if existing_user is not None:
        raise HTTPException(
            status_code=409,
            detail="Email already registered",
        )

    # Tạo tài khoản với password đã hash
    user = User(
        email=email,
        password_hash=hash_password(
            request.password
        ),
    )

    db.add(user)

    try:
        db.commit()
        db.refresh(user)

    except IntegrityError:
        db.rollback()

        raise HTTPException(
            status_code=409,
            detail="Email already registered",
        )

    return {
        "id": user.id,
        "email": user.email,
    }


# =====================================
# LOGIN
# =====================================


@router.post(
    "/login",
    response_model=TokenResponse,
)
def login(
    request: LoginRequest,
    db: Session = Depends(get_db),
):
    email = str(request.email).strip().lower()

    user = db.scalar(
        select(User).where(User.email == email)
    )

    if user is None or not verify_password(
        request.password,
        user.password_hash,
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(user.id)

    return {
        "access_token": token,
        "token_type": "bearer",
    }
