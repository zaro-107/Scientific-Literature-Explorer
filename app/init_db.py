from app.database import Base, engine
import app.models  # registers models with SQLAlchemy

def init_db():
    Base.metadata.create_all(bind=engine)
