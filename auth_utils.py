from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta

# 🔐 password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

SECRET_KEY = "sonixa_secret_key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60


# ✅ HASH PASSWORD
def hash_password(password: str):
    password = password[:72]  # bcrypt fix
    return pwd_context.hash(password)


# ✅ VERIFY PASSWORD
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password[:72], hashed_password)


# ✅ CREATE TOKEN
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    return encoded_jwt


# ✅ DECODE TOKEN
def decode_access_token(token: str):
    try:
        decoded = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return decoded
    except JWTError:
        return None