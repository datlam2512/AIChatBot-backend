
from dotenv import dotenv_values
from sqlalchemy import create_engine, text

# Đọc cấu hình Supabase riêng
config = dotenv_values(".env.cloud.local")

database_url = config["DATABASE_URL"]

# Tạo kết nối đến PostgreSQL cloud
engine = create_engine(database_url)

try:
    with engine.connect() as connection:
        # Kiểm tra database
        result = connection.execute(
            text("SELECT current_database(), current_user")
        )

        print("Database:", result.one())

        # Kiểm tra dữ liệu đã migration
        tables = [
            "users",
            "conversations",
            "messages",
            "documents",
            "document_chunks",
            "checkpoints",
        ]

        for table in tables:
            result = connection.execute(
                text(f"SELECT COUNT(*) FROM public.{table}")
            )

            print(f"{table}: {result.scalar_one()}")

    print("Supabase connection successful!")

finally:
    engine.dispose()
