from requests import get
from bs4 import BeautifulSoup as soupify


def get_blog_urls(base_url, header={'User-Agent': 'hamsandwich'}):
    soup = soupify(get(base_url, headers=header).content)
    return [link['href'] for link in soup.select('a.more-link')], header


def get_blog_content(base_url):
    blog_links, header = get_blog_urls(base_url)
    all_blogs = []
    for blog in blog_links:
        blog_soup = soupify(
            get(blog,
                headers=header).content)
        blog_content = {'title': blog_soup.select_one(
            'h1.entry-title').text,
        'content': blog_soup.select_one(
            'div.entry-content').text.strip()}
        all_blogs.append(blog_content)
    return all_blogs


def get_cats(base_url):
    soup = soupify(get(base_url).content)
    return [cat.text.lower() for cat in soup.find_all('li')[1:]]

def get_all_shorts(base_url):
    cats = get_cats(base_url)
    all_articles = []
    for cat in cats:
        cat_url = base_url + '/' + cat
        print(get(cat_url))
        cat_soup = soupify(get(cat_url).content)
        cat_titles = [
            title.text for title in cat_soup.find_all('span', itemprop='headline')
        ]
        cat_bodies = [
            body.text for body in cat_soup.find_all('div', itemprop='articleBody')]
        cat_articles = [{'title': title,
        'category': cat,
        'body': body} for title, body in zip(
        cat_titles, cat_bodies)]
        print('cat articles length: ',len(cat_articles))
        all_articles.extend(cat_articles)
        print('length of all_articles: ', len(all_articles))
    return all_articles