import re
import requests
from bs4 import BeautifulSoup as BFS
import pandas as pd

url = "https://food.trueid.net/detail/2Q6J46VdqKAQ"
webpage = requests.get(url)
soup = BFS(webpage.content, "html.parser")

stores = soup.find_all("h3", style="text-align:center")
store_names = []
store_ranks = []
ranking = 1
for item in stores:
    if len(item.text) > 1:
        store_ranks.append(ranking)
        item_name = item.text.split(" ", 1)[1]
        store_names.append(item_name)
        ranking += 1

# Initialize parameters
default_value = "ไม่มี"
lst_coordinates = []
lst_address = []
lst_opening_hours = []
lst_phone = []
lst_parking = []
lst_website = []
ul_elements = soup.find_all("ul")
for ul in ul_elements:
    if len(ul) > 4:
        coordinates = default_value
        address = default_value
        opening_hours = default_value
        phone = default_value
        parking = default_value
        website = default_value
        li_elements = ul.find_all("li")
        for li in li_elements:
            text = li.get_text(strip=True)
            if text.startswith("พิกัด"):
                coordinates = li.find("a")["href"]
            elif text.startswith("ที่อยู่"):
                address = text.split(":")[1].strip()
            elif text.startswith("เปิดบริการ"):
                opening_hours = text.split(":")[1].strip()
            elif text.startswith("โทร"):
                phone = text.split(":")[1].strip()
            elif text.startswith("ที่จอดรถ"):
                parking = text.split(":")[1].strip()
            elif text.startswith("เว็บไซต์"):
                website = li.find("a")["href"]
        lst_coordinates.append(coordinates)
        lst_address.append(address)
        lst_opening_hours.append(opening_hours)
        lst_phone.append(phone)
        lst_parking.append(parking)
        lst_website.append(website)

hangout_dict = {
    "อันดับ": store_ranks,
    "ชื่อร้าน": store_names,
    "พิกัด": lst_coordinates,
    "ที่อยู่": lst_address,
    "เวลาทำการ": lst_opening_hours,
    "ช่องทางติดต่อ": lst_phone,
    "ที่จอดรถ": lst_parking,
    "เว็บไซต์": lst_website,
}
hangout_df = pd.DataFrame(hangout_dict)

# Filtering opening hour
hangout_time = hangout_df["เวลาทำการ"]
conditions = []
for i in range(len(hangout_time)):
    try:
        hour = int(list(hangout_time)[i][8:10])
        if hour < 10:
            condition = "ใช่"
        else:
            condition = "ไม่ใช่"
    except:
        condition = "ไม่ทราบ"
    conditions.append(condition)
hangout_df["เปิดหลังเที่ยงคืน"] = conditions

# Filtering Parking
parking_pattern = r"(ไม่มี|มี)"
hangout_parking = hangout_df["ที่จอดรถ"].tolist()
hangout_parking = [
    re.search(parking_pattern, p).group() if re.search(parking_pattern, p) else "ไม่ทราบ"
    for p in hangout_parking
]
replacement_mapping = {"ไม่มี": "ไม่ใช่", "มี": "ใช่"}
converted_matches = [replacement_mapping.get(match, match) for match in hangout_parking]
hangout_df["มีที่จอดรถ"] = converted_matches

# Save to CSV
hangout_df.to_csv("hangout_info.csv", index=False)
print("Saved hangout_info.csv")
