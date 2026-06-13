def calculate_pm25_aqi(pm25: float) -> int:
    if pm25 <= 30:
        return round((pm25 / 30) * 50)
    if pm25 <= 60:
        return round(((pm25 - 31) / (60 - 31)) * (100 - 51) + 51)
    if pm25 <= 90:
        return round(((pm25 - 61) / (90 - 61)) * (200 - 101) + 101)
    if pm25 <= 120:
        return round(((pm25 - 91) / (120 - 91)) * (300 - 201) + 201)
    if pm25 <= 250:
        return round(((pm25 - 121) / (250 - 121)) * (400 - 301) + 301)
    return round(((pm25 - 251) / (500 - 251)) * (500 - 401) + 401)


def aqi_category(aqi: int) -> str:
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Satisfactory"
    if aqi <= 200:
        return "Moderate"
    if aqi <= 300:
        return "Poor"
    if aqi <= 400:
        return "Very Poor"
    return "Severe"
