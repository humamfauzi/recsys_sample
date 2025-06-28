from enum import Enum

# Note this is for raw data
class Users(Enum):
    """Enum for user-related mappings."""
    1 = "id"
    2 = "age"
    3 = "gender"
    4 = "occupation"
    5 = "zip_code"

class Products(Enum):
    """Enum for product-related mappings."""
    1 = "id"
    2 = "title"
    3 = "release_date"
    4 = "video_release_date"
    5 = "imdb_url" 
    6 = "unknown"
    7 = "Action"
    8 = "Adventure"
    9 = "Animation"
    10 = "Children"
    11 = "Comedy"
    12 = "Crime"
    13 = "Documentary"
    14 = "Drama"
    15 = "Fantasy"
    16 = "Film-Noir"
    17 = "Horror"
    18 = "Musical"
    19 = "Mystery"
    20 = "Romance"
    21 = "Sci-Fi"
    22 = "Thriller"
    23 = "War"
    24 = "Western"

class Ratings(Enum):
    """Enum for rating-related mappings."""
    1 = "user_id"
    2 = "product_id"
    3 = "rating"