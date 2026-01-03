import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import os
import time

PAPERS_TO_FIND = [
    {
        "query": "Perception Tokens Enhance Visual Reasoning",
        "filename": "2024_Bigverdi_PerceptionTokens.pdf",
        "title_match": "Perception Tokens"
    },
    {
        "query": "Dynamic Visual Reasoning by Learning Differentiable Physics Models",
        "filename": "2021_Ding_DynamicVisualReasoning.pdf",
        "title_match": "Dynamic Visual Reasoning"
    },
    {
        "query": "Object-Centric Learning with Slot Attention",
        "filename": "2020_Locatello_SlotAttention.pdf",
        "title_match": "Slot Attention"
    },
    {
        "query": "Recurrent Independent Mechanisms",
        "filename": "2019_Goyal_RIM.pdf",
        "title_match": "Recurrent Independent Mechanisms"
    },
    {
        "query": "Machine Mental Imagery: Empower Multimodal Reasoning with Latent Visual Tokens",
        "filename": "2025_Luo_Mirage.pdf",
        "title_match": "Machine Mental Imagery"
    }
]

def search_and_download(paper_info):
    base_url = 'http://export.arxiv.org/api/query?'
    search_query = urllib.parse.quote(paper_info["query"])
    url = f'{base_url}search_query=all:{search_query}&start=0&max_results=1'
    
    try:
        data = urllib.request.urlopen(url).read()
        root = ET.fromstring(data)
        entry = root.find('{http://www.w3.org/2005/Atom}entry')
        
        if entry is None:
            print(f"No results for: {paper_info['query']}")
            return None

        title = entry.find('{http://www.w3.org/2005/Atom}title').text
        summary = entry.find('{http://www.w3.org/2005/Atom}summary').text
        id_url = entry.find('{http://www.w3.org/2005/Atom}id').text
        pdf_url = id_url.replace('/abs/', '/pdf/')
        authors = [a.find('{http://www.w3.org/2005/Atom}name').text for a in entry.findall('{http://www.w3.org/2005/Atom}author')]
        published = entry.find('{http://www.w3.org/2005/Atom}published').text[:4]

        print(f"Found: {title} ({published})")
        
        # Download
        print(f"Downloading to papers/{paper_info['filename']}...")
        urllib.request.urlretrieve(pdf_url, f"papers/{paper_info['filename']}")
        
        return {
            "title": title.replace('\n', ' ').strip(),
            "authors": ", ".join(authors),
            "year": published,
            "filename": paper_info["filename"],
            "url": id_url,
            "summary": summary.replace('\n', ' ').strip()
        }

    except Exception as e:
        print(f"Error processing {paper_info['query']}: {e}")
        return None

def main():
    if not os.path.exists('papers'):
        os.makedirs('papers')
        
    readme_content = "# Downloaded Papers\n\n"
    
    for paper in PAPERS_TO_FIND:
        info = search_and_download(paper)
        if info:
            readme_content += f"## [{info['title']}]({info['filename']})\n"
            readme_content += f"- **Authors**: {info['authors']}\n"
            readme_content += f"- **Year**: {info['year']}\n"
            readme_content += f"- **ArXiv**: {info['url']}\n"
            readme_content += f"- **Abstract**: {info['summary'][:300]}...\n\n"
        time.sleep(3) # Be nice to ArXiv API

    with open('papers/README.md', 'w') as f:
        f.write(readme_content)

if __name__ == "__main__":
    main()
