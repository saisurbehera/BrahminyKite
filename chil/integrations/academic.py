"""
Academic Database Integrations

Provides real connections to academic databases like PubMed, ArXiv, CrossRef, and Semantic Scholar.
"""

import aiohttp
import asyncio
import logging
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from urllib.parse import quote
import re


@dataclass
class AcademicPaper:
    """Standardized academic paper representation"""
    title: str
    authors: List[str]
    abstract: str
    journal: Optional[str] = None
    year: Optional[int] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None  # PubMed ID
    arxiv_id: Optional[str] = None
    url: Optional[str] = None
    citations: int = 0
    keywords: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    
    def get_citation_string(self) -> str:
        """Get formatted citation"""
        author_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            author_str += " et al."
        
        year_str = f" ({self.year})" if self.year else ""
        journal_str = f" {self.journal}." if self.journal else ""
        doi_str = f" DOI: {self.doi}" if self.doi else ""
        
        return f"{author_str}{year_str}. {self.title}.{journal_str}{doi_str}"


class AcademicAPI(ABC):
    """Abstract base class for academic database APIs"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    @abstractmethod
    async def search(self, query: str, max_results: int = 10) -> List[AcademicPaper]:
        """Search for papers matching query"""
        pass
    
    @abstractmethod
    async def get_paper(self, paper_id: str) -> Optional[AcademicPaper]:
        """Get specific paper by ID"""
        pass
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing extra whitespace and special characters"""
        if not text:
            return ""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove common HTML entities
        text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        return text.strip()


class PubMedAPI(AcademicAPI):
    """PubMed/MEDLINE database integration"""
    
    def __init__(self, email: Optional[str] = None):
        super().__init__()
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.email = email  # NCBI requests email for heavy usage
    
    async def search(self, query: str, max_results: int = 10) -> List[AcademicPaper]:
        """Search PubMed for papers"""
        try:
            # First, search for IDs
            search_url = f"{self.base_url}/esearch.fcgi"
            params = {
                'db': 'pubmed',
                'term': query,
                'retmax': max_results,
                'retmode': 'json',
                'sort': 'relevance'
            }
            
            if self.email:
                params['email'] = self.email
            
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.get(search_url, params=params) as response:
                if response.status != 200:
                    self.logger.error(f"PubMed search failed: {response.status}")
                    return []
                
                data = await response.json()
                id_list = data.get('esearchresult', {}).get('idlist', [])
                
                if not id_list:
                    return []
                
                # Fetch details for each ID
                return await self._fetch_papers(id_list)
                
        except Exception as e:
            self.logger.error(f"PubMed search error: {e}")
            return []
    
    async def _fetch_papers(self, pmids: List[str]) -> List[AcademicPaper]:
        """Fetch paper details for given PMIDs"""
        fetch_url = f"{self.base_url}/efetch.fcgi"
        params = {
            'db': 'pubmed',
            'id': ','.join(pmids),
            'retmode': 'xml',
            'rettype': 'abstract'
        }
        
        if self.email:
            params['email'] = self.email
        
        try:
            async with self.session.get(fetch_url, params=params) as response:
                if response.status != 200:
                    return []
                
                xml_data = await response.text()
                return self._parse_pubmed_xml(xml_data)
                
        except Exception as e:
            self.logger.error(f"PubMed fetch error: {e}")
            return []
    
    def _parse_pubmed_xml(self, xml_data: str) -> List[AcademicPaper]:
        """Parse PubMed XML response"""
        papers = []
        
        try:
            root = ET.fromstring(xml_data)
            
            for article in root.findall('.//PubmedArticle'):
                medline = article.find('.//MedlineCitation')
                if not medline:
                    continue
                
                # Extract PMID
                pmid_elem = medline.find('.//PMID')
                pmid = pmid_elem.text if pmid_elem is not None else None
                
                # Extract article info
                article_elem = medline.find('.//Article')
                if not article_elem:
                    continue
                
                # Title
                title_elem = article_elem.find('.//ArticleTitle')
                title = self._clean_text(title_elem.text) if title_elem is not None else "Unknown Title"
                
                # Authors
                authors = []
                for author in article_elem.findall('.//Author'):
                    last_name = author.find('.//LastName')
                    first_name = author.find('.//ForeName')
                    if last_name is not None:
                        name = last_name.text
                        if first_name is not None:
                            name = f"{first_name.text} {name}"
                        authors.append(name)
                
                # Abstract
                abstract_elem = article_elem.find('.//Abstract/AbstractText')
                abstract = self._clean_text(abstract_elem.text) if abstract_elem is not None else ""
                
                # Journal
                journal_elem = article_elem.find('.//Journal/Title')
                journal = journal_elem.text if journal_elem is not None else None
                
                # Year
                year_elem = article_elem.find('.//Journal/JournalIssue/PubDate/Year')
                year = int(year_elem.text) if year_elem is not None and year_elem.text else None
                
                # DOI
                doi_elem = article.find('.//ArticleIdList/ArticleId[@IdType="doi"]')
                doi = doi_elem.text if doi_elem is not None else None
                
                # Keywords
                keywords = []
                for keyword in article_elem.findall('.//Keyword'):
                    if keyword.text:
                        keywords.append(keyword.text)
                
                # Create paper object
                paper = AcademicPaper(
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    journal=journal,
                    year=year,
                    doi=doi,
                    pmid=pmid,
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    keywords=keywords
                )
                
                papers.append(paper)
                
        except ET.ParseError as e:
            self.logger.error(f"XML parsing error: {e}")
        
        return papers
    
    async def get_paper(self, pmid: str) -> Optional[AcademicPaper]:
        """Get specific paper by PMID"""
        papers = await self._fetch_papers([pmid])
        return papers[0] if papers else None


class ArXivAPI(AcademicAPI):
    """arXiv preprint repository integration"""
    
    def __init__(self):
        super().__init__()
        self.base_url = "http://export.arxiv.org/api/query"
    
    async def search(self, query: str, max_results: int = 10) -> List[AcademicPaper]:
        """Search arXiv for papers"""
        try:
            params = {
                'search_query': f'all:{query}',
                'start': 0,
                'max_results': max_results,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.get(self.base_url, params=params) as response:
                if response.status != 200:
                    self.logger.error(f"arXiv search failed: {response.status}")
                    return []
                
                xml_data = await response.text()
                return self._parse_arxiv_xml(xml_data)
                
        except Exception as e:
            self.logger.error(f"arXiv search error: {e}")
            return []
    
    def _parse_arxiv_xml(self, xml_data: str) -> List[AcademicPaper]:
        """Parse arXiv XML response"""
        papers = []
        
        try:
            # Parse XML with namespace
            root = ET.fromstring(xml_data)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            for entry in root.findall('atom:entry', ns):
                # Title
                title_elem = entry.find('atom:title', ns)
                title = self._clean_text(title_elem.text) if title_elem is not None else "Unknown Title"
                
                # Authors
                authors = []
                for author in entry.findall('atom:author', ns):
                    name_elem = author.find('atom:name', ns)
                    if name_elem is not None and name_elem.text:
                        authors.append(name_elem.text.strip())
                
                # Abstract
                summary_elem = entry.find('atom:summary', ns)
                abstract = self._clean_text(summary_elem.text) if summary_elem is not None else ""
                
                # arXiv ID
                id_elem = entry.find('atom:id', ns)
                arxiv_id = None
                url = None
                if id_elem is not None and id_elem.text:
                    url = id_elem.text
                    # Extract ID from URL
                    match = re.search(r'arxiv.org/abs/(\d+\.\d+)', url)
                    if match:
                        arxiv_id = match.group(1)
                
                # Published date
                published_elem = entry.find('atom:published', ns)
                year = None
                if published_elem is not None and published_elem.text:
                    try:
                        year = int(published_elem.text[:4])
                    except (ValueError, IndexError):
                        pass
                
                # DOI (if available)
                doi = None
                for link in entry.findall('atom:link', ns):
                    if link.get('title') == 'doi':
                        doi = link.get('href', '').replace('http://dx.doi.org/', '')
                
                # Categories as keywords
                keywords = []
                for category in entry.findall('atom:category', ns):
                    term = category.get('term')
                    if term:
                        keywords.append(term)
                
                # Create paper object
                paper = AcademicPaper(
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    journal="arXiv preprint",
                    year=year,
                    doi=doi,
                    arxiv_id=arxiv_id,
                    url=url,
                    keywords=keywords
                )
                
                papers.append(paper)
                
        except ET.ParseError as e:
            self.logger.error(f"XML parsing error: {e}")
        
        return papers
    
    async def get_paper(self, arxiv_id: str) -> Optional[AcademicPaper]:
        """Get specific paper by arXiv ID"""
        params = {
            'id_list': arxiv_id,
            'max_results': 1
        }
        
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
                
            async with self.session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    xml_data = await response.text()
                    papers = self._parse_arxiv_xml(xml_data)
                    return papers[0] if papers else None
        except Exception as e:
            self.logger.error(f"arXiv fetch error: {e}")
        
        return None


class CrossRefAPI(AcademicAPI):
    """CrossRef database integration"""
    
    def __init__(self, email: Optional[str] = None):
        super().__init__()
        self.base_url = "https://api.crossref.org"
        self.email = email  # Polite API usage
    
    async def search(self, query: str, max_results: int = 10) -> List[AcademicPaper]:
        """Search CrossRef for papers"""
        try:
            url = f"{self.base_url}/works"
            params = {
                'query': query,
                'rows': max_results,
                'sort': 'relevance',
                'order': 'desc'
            }
            
            headers = {}
            if self.email:
                headers['User-Agent'] = f"BrahminyKite/1.0 (mailto:{self.email})"
            
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status != 200:
                    self.logger.error(f"CrossRef search failed: {response.status}")
                    return []
                
                data = await response.json()
                return self._parse_crossref_response(data)
                
        except Exception as e:
            self.logger.error(f"CrossRef search error: {e}")
            return []
    
    def _parse_crossref_response(self, data: Dict[str, Any]) -> List[AcademicPaper]:
        """Parse CrossRef JSON response"""
        papers = []
        
        items = data.get('message', {}).get('items', [])
        
        for item in items:
            # Title
            title_list = item.get('title', [])
            title = title_list[0] if title_list else "Unknown Title"
            
            # Authors
            authors = []
            for author in item.get('author', []):
                given = author.get('given', '')
                family = author.get('family', '')
                if family:
                    name = f"{given} {family}".strip() if given else family
                    authors.append(name)
            
            # Abstract
            abstract = item.get('abstract', '')
            
            # Journal
            container_title = item.get('container-title', [])
            journal = container_title[0] if container_title else None
            
            # Year
            date_parts = item.get('published-print', {}).get('date-parts', [[]])
            if not date_parts:
                date_parts = item.get('published-online', {}).get('date-parts', [[]])
            year = date_parts[0][0] if date_parts and date_parts[0] else None
            
            # DOI
            doi = item.get('DOI')
            
            # URL
            url = item.get('URL') or (f"https://doi.org/{doi}" if doi else None)
            
            # Citations (reference count as proxy)
            citations = item.get('is-referenced-by-count', 0)
            
            # Create paper object
            paper = AcademicPaper(
                title=self._clean_text(title),
                authors=authors,
                abstract=self._clean_text(abstract),
                journal=journal,
                year=year,
                doi=doi,
                url=url,
                citations=citations
            )
            
            papers.append(paper)
        
        return papers
    
    async def get_paper(self, doi: str) -> Optional[AcademicPaper]:
        """Get specific paper by DOI"""
        try:
            url = f"{self.base_url}/works/{doi}"
            
            headers = {}
            if self.email:
                headers['User-Agent'] = f"BrahminyKite/1.0 (mailto:{self.email})"
            
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    papers = self._parse_crossref_response({'message': {'items': [data['message']]}})
                    return papers[0] if papers else None
                    
        except Exception as e:
            self.logger.error(f"CrossRef fetch error: {e}")
        
        return None


class SemanticScholarAPI(AcademicAPI):
    """Semantic Scholar integration"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.api_key = api_key
    
    async def search(self, query: str, max_results: int = 10) -> List[AcademicPaper]:
        """Search Semantic Scholar for papers"""
        try:
            url = f"{self.base_url}/paper/search"
            params = {
                'query': query,
                'limit': max_results,
                'fields': 'title,authors,abstract,year,citationCount,journal,doi,url,arxivId,pubMedId'
            }
            
            headers = {}
            if self.api_key:
                headers['x-api-key'] = self.api_key
            
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status != 200:
                    self.logger.error(f"Semantic Scholar search failed: {response.status}")
                    return []
                
                data = await response.json()
                return self._parse_s2_response(data)
                
        except Exception as e:
            self.logger.error(f"Semantic Scholar search error: {e}")
            return []
    
    def _parse_s2_response(self, data: Dict[str, Any]) -> List[AcademicPaper]:
        """Parse Semantic Scholar response"""
        papers = []
        
        for item in data.get('data', []):
            # Authors
            authors = []
            for author in item.get('authors', []):
                if author.get('name'):
                    authors.append(author['name'])
            
            # Journal info
            journal_info = item.get('journal', {})
            journal = journal_info.get('name')
            
            # Create paper object
            paper = AcademicPaper(
                title=item.get('title', 'Unknown Title'),
                authors=authors,
                abstract=item.get('abstract', ''),
                journal=journal,
                year=item.get('year'),
                doi=item.get('doi'),
                pmid=item.get('pubMedId'),
                arxiv_id=item.get('arxivId'),
                url=item.get('url'),
                citations=item.get('citationCount', 0)
            )
            
            papers.append(paper)
        
        return papers
    
    async def get_paper(self, paper_id: str) -> Optional[AcademicPaper]:
        """Get specific paper by Semantic Scholar ID, DOI, or other identifier"""
        try:
            # Try different ID formats
            if paper_id.startswith('10.'):  # Likely a DOI
                url = f"{self.base_url}/paper/DOI:{paper_id}"
            elif paper_id.isdigit():  # Likely a PMID
                url = f"{self.base_url}/paper/PMID:{paper_id}"
            elif '.' in paper_id and paper_id.replace('.', '').isdigit():  # Likely arXiv
                url = f"{self.base_url}/paper/ARXIV:{paper_id}"
            else:
                url = f"{self.base_url}/paper/{paper_id}"
            
            params = {
                'fields': 'title,authors,abstract,year,citationCount,journal,doi,url,arxivId,pubMedId'
            }
            
            headers = {}
            if self.api_key:
                headers['x-api-key'] = self.api_key
            
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    papers = self._parse_s2_response({'data': [data]})
                    return papers[0] if papers else None
                    
        except Exception as e:
            self.logger.error(f"Semantic Scholar fetch error: {e}")
        
        return None


class AcademicHub:
    """Hub for coordinating multiple academic database searches"""
    
    def __init__(self, email: Optional[str] = None, api_keys: Optional[Dict[str, str]] = None):
        self.email = email
        self.api_keys = api_keys or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize APIs
        self.apis = {
            'pubmed': PubMedAPI(email),
            'arxiv': ArXivAPI(),
            'crossref': CrossRefAPI(email),
            'semantic_scholar': SemanticScholarAPI(self.api_keys.get('semantic_scholar'))
        }
    
    async def search_all(self, query: str, max_results_per_source: int = 5) -> Dict[str, List[AcademicPaper]]:
        """Search all databases concurrently"""
        results = {}
        
        # Create tasks for all APIs
        tasks = []
        for name, api in self.apis.items():
            task = asyncio.create_task(self._search_with_context(api, query, max_results_per_source))
            tasks.append((name, task))
        
        # Wait for all searches to complete
        for name, task in tasks:
            try:
                papers = await task
                results[name] = papers
            except Exception as e:
                self.logger.error(f"Search error for {name}: {e}")
                results[name] = []
        
        return results
    
    async def _search_with_context(self, api: AcademicAPI, query: str, max_results: int) -> List[AcademicPaper]:
        """Search with context manager"""
        async with api:
            return await api.search(query, max_results)
    
    async def find_similar_papers(self, title: str, max_results: int = 10) -> List[AcademicPaper]:
        """Find papers similar to given title across all databases"""
        all_papers = []
        
        results = await self.search_all(title, max_results_per_source=max_results)
        
        # Combine and deduplicate results
        seen_titles = set()
        for source_papers in results.values():
            for paper in source_papers:
                # Simple deduplication by normalized title
                normalized_title = paper.title.lower().strip()
                if normalized_title not in seen_titles:
                    seen_titles.add(normalized_title)
                    all_papers.append(paper)
        
        # Sort by relevance (citation count as proxy)
        all_papers.sort(key=lambda p: p.citations, reverse=True)
        
        return all_papers[:max_results]
    
    def filter_papers_by_year(self, papers: List[AcademicPaper], min_year: int) -> List[AcademicPaper]:
        """Filter papers by minimum year"""
        return [p for p in papers if p.year and p.year >= min_year]
    
    def filter_papers_by_keywords(self, papers: List[AcademicPaper], keywords: List[str]) -> List[AcademicPaper]:
        """Filter papers that contain any of the keywords in title or abstract"""
        filtered = []
        keywords_lower = [k.lower() for k in keywords]
        
        for paper in papers:
            text = f"{paper.title} {paper.abstract}".lower()
            if any(keyword in text for keyword in keywords_lower):
                filtered.append(paper)
        
        return filtered