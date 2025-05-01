-- Add YouTube URL related columns if they don't exist
DO $$ 
BEGIN
    -- Add youtube_url_updated_at column if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                  WHERE table_name='songs' AND column_name='youtube_url_updated_at') THEN
        ALTER TABLE songs ADD COLUMN youtube_url_updated_at TIMESTAMP;
    END IF;

    -- Add youtube_url_status column if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                  WHERE table_name='songs' AND column_name='youtube_url_status') THEN
        ALTER TABLE songs ADD COLUMN youtube_url_status VARCHAR;
    END IF;

    -- Add youtube_url_error column if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                  WHERE table_name='songs' AND column_name='youtube_url_error') THEN
        ALTER TABLE songs ADD COLUMN youtube_url_error TEXT;
    END IF;
END $$; 