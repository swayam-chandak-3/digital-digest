# Fix for Python 3.9+ compatibility with older BeautifulSoup versions
# Must be at the very top before any other imports
import sys
import collections
if sys.version_info >= (3, 9):
    try:
        from collections.abc import Callable
        if not hasattr(collections, 'Callable'):
            collections.Callable = Callable
    except ImportError:
        from typing import Callable
        if not hasattr(collections, 'Callable'):
            collections.Callable = Callable

import requests
import os
from multiprocessing import Pool, cpu_count
from functools import partial
from bs4 import BeautifulSoup, SoupStrainer
#Makes Output Directory if it does not exist
if not os.path.exists(os.path.join(os.getcwd(), 'HackerNews')):
  os.makedirs(os.path.join(os.getcwd(), 'HackerNews'))
'''
@params page_no: The page number of HackerNews to fetch.
Adding only page number in order to add multiprocess support in future.
@params verbose: Adds verbose output to screen instead of running the program silently.
'''
def fetch(page_no, verbose=False):
    #Should be unreachable, but just in case
    if page_no <= 0:
        raise ValueError('Number of Pages must be greater than zero')
    page_no = min(page_no, 20)
    i = page_no
    if verbose:
        print('Fetching Page {}...'.format(i))
    try:
        res = requests.get('https://news.ycombinator.com/?p='+str(i))
        only_td = SoupStrainer('td')
        soup = BeautifulSoup(res.content, 'html.parser', parse_only=only_td)
        tdtitle = soup.find_all('td', attrs={'class':'title'})
        tdmetrics = soup.find_all('td', attrs={'class':'subtext'})
        page_urls = []  # Store URLs for this page
        with open(os.path.join('HackerNews', 'NewsPage{}.txt'.format(i)), 'w+') as f:
            f.write('-'*80)
            f.write('\n')
            f.write('Page {}'.format(i))
            tdtitle = soup.find_all('td', attrs={'class':'title'})
            tdrank = soup.find_all('td', attrs={'class':'title', 'align':'right'})
            tdtitleonly = [t for t in tdtitle if t not in tdrank]
            tdmetrics = soup.find_all('td', attrs={'class':'subtext'})
            tdt = tdtitleonly
            tdr = tdrank
            tdm = tdmetrics
            num_iter = min(len(tdr), len(tdt))
            for idx in range(num_iter):
                f.write('\n'+'-'*80+'\n')
                rank = tdr[idx].find('span', attrs={'class':'rank'})
                # Title link is now inside <span class="titleline">
                titleline = tdt[idx].find('span', attrs={'class':'titleline'})
                titl = titleline.find('a') if titleline else None
                if titl and 'href' in titl.attrs:
                    url = titl['href'] if titl['href'].startswith('https') or titl['href'].startswith('http') else 'https://news.ycombinator.com/'+titl['href']
                    page_urls.append(url)  # Collect URL
                else:
                    url = None
                # Site info is now in <span class="sitebit comhead">
                site = tdt[idx].find('span', attrs={'class':'sitebit comhead'})
                score = tdm[idx].find('span', attrs={'class':'score'})
                time = tdm[idx].find('span', attrs={'class':'age'})
                author = tdm[idx].find('a', attrs={'class':'hnuser'})
                f.write('\nArticle Number: '+rank.text.replace('.','') if rank else '\nArticle Number: Could not get article number')
                f.write('\nArticle Title: '+titl.text if titl else '\nArticle Title: Could not get article title')
                f.write('\nSource Website: '+site.text if site else '\nSource Website: https://news.ycombinator.com')
                f.write('\nSource URL: '+url if url else '\nSource URL: No URL found for this article')
                f.write('\nArticle Author: '+author.text if author else '\nArticle Author: Could not get article author')
                f.write('\nArticle Score: '+score.text if score else '\nArticle Score: Not Scored')
                f.write('\nPosted: '+time.text if time else '\nPosted: Could not find when the article was posted')
                f.write('\n'+'-'*80+'\n')
        
        # Save URLs to a separate file for this page
        if page_urls:
            urls_file = os.path.join('HackerNews', 'NewsPage{}_urls.txt'.format(i))
            with open(urls_file, 'w', encoding='utf-8') as uf:
                for url in page_urls:
                    uf.write(url + '\n')
            if verbose:
                print(f'  Saved {len(page_urls)} URLs to {urls_file}')
    except (requests.ConnectionError, requests.packages.urllib3.exceptions.ConnectionError) as e:
        print('Connection Failed for page {}'.format(i))
    except requests.RequestException as e:
        print("Some ambiguous Request Exception occurred. The exception is "+str(e))
while(True):
    try:
        pages = int(input('Enter number of pages that you want the HackerNews for (max 20): '))
        v = input('Want verbose output y/[n] ?')
        verbose = v.lower().startswith('y')
        if pages > 20:
            print('A maximum of only 20 pages can be fetched')
        pages = min(pages, 20)
        for page_no in range(1, pages + 1):
            fetch(page_no, verbose)
        break
    except ValueError as e:
        print('\nInvalid input, probably not a positive integer\n')
        continue
