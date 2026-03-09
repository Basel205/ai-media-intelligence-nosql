from database.schema_setup import SchemaSetup


def main():
    print("Initializing database...")
    schema = SchemaSetup()
    schema.setup()
    print("Database ready.")


if __name__ == "__main__":
    main()