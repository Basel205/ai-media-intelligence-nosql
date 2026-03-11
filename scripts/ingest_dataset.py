from ingestion.pipeline import MediaIngestionPipeline


def main():

    media_folder = "data/raw_media"

    pipeline = MediaIngestionPipeline(media_folder)

    pipeline.run()


if __name__ == "__main__":
    main()