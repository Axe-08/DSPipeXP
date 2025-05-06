# Setting Up a Local Database for DSPipeXP

This guide will help you set up a local PostgreSQL database for running the DSPipeXP Music Recommendation System on your computer.

## Prerequisites

- PostgreSQL 12 or newer installed on your computer
- Python 3.8 or newer
- The DSPipeXP repository cloned to your computer

## Step 1: Install PostgreSQL

If you haven't installed PostgreSQL yet, follow these instructions:

### Windows
1. Download the installer from [PostgreSQL.org](https://www.postgresql.org/download/windows/)
2. Run the installer and follow the prompts
3. Remember the password you set for the 'postgres' user
4. Add PostgreSQL bin directory to your PATH (the installer should offer this option)

### macOS
Using Homebrew:
```bash
brew install postgresql
brew services start postgresql
```

### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo service postgresql start
```

## Step 2: Set Up Database

We've created a script that will automatically set up your database with the initial dataset. The script will:

1. Create a new PostgreSQL database
2. Set up the necessary tables
3. Import song data from the `spotify_songs.csv` dataset in the `lyric_dataset` folder
4. Configure indexes for better performance

### Run the Setup Script

```bash
# Navigate to the project root
cd /path/to/DSPipeXP

# Install required Python packages
pip install pandas psycopg2-binary sqlalchemy

# Run the database setup script
python scripts/init_local_db.py --db-user your_postgres_username --db-password your_postgres_password
```

Replace `your_postgres_username` and `your_postgres_password` with your PostgreSQL credentials. By default, this is often `postgres` for the username.

### Additional Options

The script supports several optional parameters:

- `--db-name`: Name for your database (default: "dspipexp")
- `--db-host`: Database host address (default: "localhost")
- `--db-port`: Database port (default: 5432)
- `--force-recreate`: Force recreation of the database if it already exists

Example with all options:
```bash
python scripts/init_local_db.py --db-user postgres --db-password mypassword --db-name music_app --db-host localhost --db-port 5432 --force-recreate
```

## Step 3: Configure Application to Use Your Database

Once your database is set up, you need to tell the application how to connect to it:

1. Create a `.streamlit` directory in the project root if it doesn't exist
2. Create a file `.streamlit/secrets.toml` with the following content:

```toml
# Database Configuration
db_url = "postgresql://your_username:your_password@localhost:5432/dspipexp"

# Optional API keys
[api_keys]
genius_access_token = "your_genius_token_here"  # Optional, for lyrics fetching
```

Replace the database URL with your actual connection details.

## Step 4: Run the Application

Now you can run the Streamlit application:

```bash
cd dspipexp_streamlit
streamlit run app.py
```

The application should connect to your local database and work with all features.

## Troubleshooting

### Connection Issues

If you encounter database connection issues:

1. **Check credentials**: Ensure username, password, and database name are correct
2. **Check PostgreSQL service**: Make sure PostgreSQL is running on your system
3. **Check database existence**: Confirm the database was created using a tool like pgAdmin or psql

### Dataset Not Found

If the script can't find the `spotify_songs.csv` file:

1. Make sure the file exists in the `lyric_dataset` folder
2. Try specifying the full path to the file when running the script:
   ```bash
   DATASET_PATH=/full/path/to/spotify_songs.csv python scripts/init_local_db.py ...
   ```

### Performance Concerns

The initial import might take a while depending on your computer's performance. For large datasets:

1. Be patient - it can take several minutes to import all the data
2. If you encounter memory issues, the script already uses batch processing to minimize memory usage

## Data Security Note

- The database you're setting up contains only public song data 
- No user data or personal information is stored
- Still, avoid using password-less PostgreSQL configurations or exposing your database to the network 