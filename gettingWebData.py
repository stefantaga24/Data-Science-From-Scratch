from bs4 import BeautifulSoup
import requests
# I put the relevant HTML file on GitHub. In order to fit
# the URL in the book I had to split it across two lines.
# Recall that whitespace-separated strings get concatenated.
url = ("https://holland2stay.com/residences.html")
html = requests.get(url).text
soup = BeautifulSoup(html, 'html5lib')
first_paragraph = soup.find('p')
print(first_paragraph.get_text())