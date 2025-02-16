import requests

def get_business_info(query, location):
    yelp_api_url = "https://api.yelp.com/v3/businesses/search"
    headers = {"Authorization": "Bearer your_yelp_api_key"}
    params = {"term": query, "location": location}
    response = requests.get(yelp_api_url, headers=headers, params=params)
    return response.json()
