"""
Fact-Checking API Integrations

Provides real connections to fact-checking services like Snopes, PolitiFact, and FactCheck.org.
"""

import aiohttp
import asyncio
import logging
import re
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from abc import ABC, abstractmethod
from urllib.parse import quote, urlencode
import hashlib
import json

# Try to import BeautifulSoup for web scraping fallback
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False


@dataclass
class FactCheckResult:
    """Standardized fact check result"""
    source: str
    claim: str
    verdict: str  # TRUE, FALSE, MIXED, UNPROVEN, etc.
    rating: float  # 0.0 to 1.0
    explanation: str
    url: Optional[str] = None
    date_checked: Optional[datetime] = None
    evidence_links: List[str] = None
    
    def __post_init__(self):
        if self.evidence_links is None:
            self.evidence_links = []


class FactCheckingAPI(ABC):
    """Abstract base class for fact-checking APIs"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.cache = {}
        self.cache_duration = timedelta(hours=24)
        
    @abstractmethod
    async def check_claim(self, claim: str) -> Optional[FactCheckResult]:
        """Check a claim and return standardized result"""
        pass
    
    def _get_cache_key(self, claim: str) -> str:
        """Generate cache key for claim"""
        return hashlib.md5(claim.lower().strip().encode()).hexdigest()
    
    def _get_from_cache(self, claim: str) -> Optional[FactCheckResult]:
        """Get result from cache if available and not expired"""
        key = self._get_cache_key(claim)
        if key in self.cache:
            result, timestamp = self.cache[key]
            if datetime.utcnow() - timestamp < self.cache_duration:
                self.logger.debug(f"Cache hit for claim: {claim[:50]}...")
                return result
        return None
    
    def _add_to_cache(self, claim: str, result: FactCheckResult):
        """Add result to cache"""
        key = self._get_cache_key(claim)
        self.cache[key] = (result, datetime.utcnow())
    
    def _normalize_rating(self, rating_text: str) -> float:
        """Convert various rating scales to 0.0-1.0"""
        rating_lower = rating_text.lower()
        
        # Truth ratings
        if any(word in rating_lower for word in ["true", "correct", "accurate", "confirmed"]):
            return 0.9
        elif any(word in rating_lower for word in ["mostly true", "mostly correct"]):
            return 0.75
        elif any(word in rating_lower for word in ["half true", "mixed", "partially"]):
            return 0.5
        elif any(word in rating_lower for word in ["mostly false", "mostly incorrect"]):
            return 0.25
        elif any(word in rating_lower for word in ["false", "incorrect", "wrong", "pants on fire"]):
            return 0.1
        elif any(word in rating_lower for word in ["unproven", "undetermined", "unclear"]):
            return 0.5
        else:
            return 0.5  # Default neutral


class SnopesAPI(FactCheckingAPI):
    """Snopes fact-checking integration"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        # Note: Snopes doesn't have a public API, so we'll use web scraping
        self.base_url = "https://www.snopes.com"
        self.search_url = f"{self.base_url}/search/"
    
    async def check_claim(self, claim: str) -> Optional[FactCheckResult]:
        """Check claim using Snopes"""
        # Check cache first
        cached = self._get_from_cache(claim)
        if cached:
            return cached
        
        try:
            # Search Snopes
            search_results = await self._search_snopes(claim)
            if not search_results:
                return None
            
            # Get the first relevant result
            for result in search_results[:3]:  # Check top 3 results
                fact_check = await self._parse_snopes_article(result['url'])
                if fact_check:
                    self._add_to_cache(claim, fact_check)
                    return fact_check
            
            return None
            
        except Exception as e:
            self.logger.error(f"Snopes API error: {e}")
            return None
    
    async def _search_snopes(self, query: str) -> List[Dict[str, str]]:
        """Search Snopes for relevant articles"""
        if not HAS_BS4:
            self.logger.warning("BeautifulSoup not available, cannot scrape Snopes")
            return []
        
        async with aiohttp.ClientSession() as session:
            # Use DuckDuckGo to search Snopes (more reliable than Snopes search)
            search_query = f"site:snopes.com {query}"
            ddg_url = f"https://html.duckduckgo.com/html/?q={quote(search_query)}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            try:
                async with session.get(ddg_url, headers=headers) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        results = []
                        for result in soup.find_all('div', class_='result__body')[:5]:
                            title_elem = result.find('a', class_='result__a')
                            if title_elem and 'snopes.com' in title_elem.get('href', ''):
                                results.append({
                                    'title': title_elem.text.strip(),
                                    'url': title_elem['href']
                                })
                        
                        return results
            except Exception as e:
                self.logger.error(f"Search error: {e}")
                
        return []
    
    async def _parse_snopes_article(self, url: str) -> Optional[FactCheckResult]:
        """Parse a Snopes fact-check article"""
        if not HAS_BS4:
            return None
        
        async with aiohttp.ClientSession() as session:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            try:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Extract rating
                        rating_elem = soup.find('div', class_='rating-label') or \
                                     soup.find('span', class_='rating-label-value')
                        
                        if not rating_elem:
                            return None
                        
                        verdict = rating_elem.text.strip()
                        
                        # Extract claim
                        claim_elem = soup.find('div', class_='claim') or \
                                    soup.find('p', class_='claim-text')
                        claim_text = claim_elem.text.strip() if claim_elem else "Unknown claim"
                        
                        # Extract explanation (first few paragraphs)
                        content_elem = soup.find('div', class_='single-body') or \
                                      soup.find('div', class_='content')
                        
                        explanation = ""
                        if content_elem:
                            paragraphs = content_elem.find_all('p')[:3]
                            explanation = " ".join(p.text.strip() for p in paragraphs)
                        
                        return FactCheckResult(
                            source="Snopes",
                            claim=claim_text[:500],  # Limit length
                            verdict=verdict,
                            rating=self._normalize_rating(verdict),
                            explanation=explanation[:1000],  # Limit length
                            url=url,
                            date_checked=datetime.utcnow()
                        )
                        
            except Exception as e:
                self.logger.error(f"Parse error for {url}: {e}")
                
        return None


class PolitiFactAPI(FactCheckingAPI):
    """PolitiFact fact-checking integration"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        # PolitiFact has a limited API through their partnership program
        self.base_url = "https://www.politifact.com/api/v2"
        self.statements_url = f"{self.base_url}/statements/"
    
    async def check_claim(self, claim: str) -> Optional[FactCheckResult]:
        """Check claim using PolitiFact"""
        # Check cache first
        cached = self._get_from_cache(claim)
        if cached:
            return cached
        
        try:
            # If we have an API key, use the API
            if self.api_key:
                return await self._check_via_api(claim)
            else:
                # Otherwise, use web scraping
                return await self._check_via_scraping(claim)
                
        except Exception as e:
            self.logger.error(f"PolitiFact API error: {e}")
            return None
    
    async def _check_via_api(self, claim: str) -> Optional[FactCheckResult]:
        """Check using PolitiFact API (requires key)"""
        async with aiohttp.ClientSession() as session:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Accept': 'application/json'
            }
            
            params = {
                'q': claim[:200],  # Limit query length
                'limit': 5
            }
            
            try:
                async with session.get(self.statements_url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get('results'):
                            # Get the most relevant result
                            statement = data['results'][0]
                            
                            return FactCheckResult(
                                source="PolitiFact",
                                claim=statement.get('statement_text', claim),
                                verdict=statement.get('ruling', {}).get('ruling', 'Unknown'),
                                rating=self._normalize_rating(statement.get('ruling', {}).get('ruling', '')),
                                explanation=statement.get('analysis', '')[:1000],
                                url=statement.get('canonical_url'),
                                date_checked=datetime.fromisoformat(statement.get('statement_date', ''))
                                if statement.get('statement_date') else datetime.utcnow()
                            )
            except Exception as e:
                self.logger.error(f"API request error: {e}")
                
        return None
    
    async def _check_via_scraping(self, claim: str) -> Optional[FactCheckResult]:
        """Check using web scraping (no API key required)"""
        if not HAS_BS4:
            self.logger.warning("BeautifulSoup not available, cannot scrape PolitiFact")
            return None
        
        # Search PolitiFact
        search_url = f"https://www.politifact.com/search/?q={quote(claim[:100])}"
        
        async with aiohttp.ClientSession() as session:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            try:
                async with session.get(search_url, headers=headers) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Find first statement result
                        statement = soup.find('article', class_='m-statement')
                        if not statement:
                            return None
                        
                        # Extract data
                        link_elem = statement.find('a', class_='m-statement__link')
                        if not link_elem:
                            return None
                        
                        # Get the full article for details
                        article_url = f"https://www.politifact.com{link_elem['href']}"
                        return await self._parse_politifact_article(article_url)
                        
            except Exception as e:
                self.logger.error(f"Scraping error: {e}")
                
        return None
    
    async def _parse_politifact_article(self, url: str) -> Optional[FactCheckResult]:
        """Parse a PolitiFact article"""
        async with aiohttp.ClientSession() as session:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            try:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Extract rating
                        meter_elem = soup.find('div', class_='m-statement-meter')
                        if not meter_elem:
                            return None
                        
                        rating_elem = meter_elem.find('picture') or meter_elem.find('img')
                        verdict = rating_elem.get('alt', 'Unknown') if rating_elem else 'Unknown'
                        
                        # Extract claim
                        claim_elem = soup.find('div', class_='m-statement__quote')
                        claim_text = claim_elem.text.strip() if claim_elem else "Unknown claim"
                        
                        # Extract explanation
                        content_elem = soup.find('article', class_='m-textblock')
                        explanation = ""
                        if content_elem:
                            paragraphs = content_elem.find_all('p')[:3]
                            explanation = " ".join(p.text.strip() for p in paragraphs)
                        
                        return FactCheckResult(
                            source="PolitiFact",
                            claim=claim_text[:500],
                            verdict=verdict,
                            rating=self._normalize_rating(verdict),
                            explanation=explanation[:1000],
                            url=url,
                            date_checked=datetime.utcnow()
                        )
                        
            except Exception as e:
                self.logger.error(f"Parse error for {url}: {e}")
                
        return None


class FactCheckOrgAPI(FactCheckingAPI):
    """FactCheck.org integration"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        self.base_url = "https://www.factcheck.org"
    
    async def check_claim(self, claim: str) -> Optional[FactCheckResult]:
        """Check claim using FactCheck.org"""
        # Check cache first
        cached = self._get_from_cache(claim)
        if cached:
            return cached
        
        try:
            # FactCheck.org doesn't have a public API, use web scraping
            return await self._check_via_scraping(claim)
                
        except Exception as e:
            self.logger.error(f"FactCheck.org error: {e}")
            return None
    
    async def _check_via_scraping(self, claim: str) -> Optional[FactCheckResult]:
        """Check using web scraping"""
        if not HAS_BS4:
            self.logger.warning("BeautifulSoup not available, cannot scrape FactCheck.org")
            return None
        
        # Search FactCheck.org
        search_url = f"{self.base_url}/?s={quote(claim[:100])}"
        
        async with aiohttp.ClientSession() as session:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            try:
                async with session.get(search_url, headers=headers) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Find first article
                        article = soup.find('article')
                        if not article:
                            return None
                        
                        # Get article link
                        link_elem = article.find('h3').find('a') if article.find('h3') else None
                        if not link_elem:
                            return None
                        
                        article_url = link_elem['href']
                        return await self._parse_factcheck_article(article_url)
                        
            except Exception as e:
                self.logger.error(f"Scraping error: {e}")
                
        return None
    
    async def _parse_factcheck_article(self, url: str) -> Optional[FactCheckResult]:
        """Parse a FactCheck.org article"""
        async with aiohttp.ClientSession() as session:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            try:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Extract title as claim
                        title_elem = soup.find('h1', class_='entry-title')
                        claim_text = title_elem.text.strip() if title_elem else "Unknown claim"
                        
                        # Extract content
                        content_elem = soup.find('div', class_='entry-content')
                        
                        # Look for verdict in content
                        verdict = "Analysis"
                        rating = 0.5  # Default neutral
                        
                        if content_elem:
                            content_text = content_elem.text.lower()
                            
                            # Simple heuristic for rating
                            if "false" in content_text[:200] or "incorrect" in content_text[:200]:
                                verdict = "False"
                                rating = 0.1
                            elif "true" in content_text[:200] or "correct" in content_text[:200]:
                                verdict = "True"
                                rating = 0.9
                            elif "misleading" in content_text[:200]:
                                verdict = "Misleading"
                                rating = 0.3
                            
                            # Get explanation from first paragraphs
                            paragraphs = content_elem.find_all('p')[:3]
                            explanation = " ".join(p.text.strip() for p in paragraphs)
                        else:
                            explanation = "No detailed analysis available"
                        
                        return FactCheckResult(
                            source="FactCheck.org",
                            claim=claim_text[:500],
                            verdict=verdict,
                            rating=rating,
                            explanation=explanation[:1000],
                            url=url,
                            date_checked=datetime.utcnow()
                        )
                        
            except Exception as e:
                self.logger.error(f"Parse error for {url}: {e}")
                
        return None


class FactCheckingHub:
    """Hub for coordinating multiple fact-checking sources"""
    
    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        self.api_keys = api_keys or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize fact checkers
        self.checkers = {
            'snopes': SnopesAPI(self.api_keys.get('snopes')),
            'politifact': PolitiFactAPI(self.api_keys.get('politifact')),
            'factcheck_org': FactCheckOrgAPI(self.api_keys.get('factcheck_org'))
        }
        
        # Weights for different sources
        self.source_weights = {
            'snopes': 0.9,
            'politifact': 0.95,
            'factcheck_org': 0.85
        }
    
    async def check_claim(self, claim: str, sources: Optional[List[str]] = None) -> List[FactCheckResult]:
        """Check claim across multiple sources"""
        if sources is None:
            sources = list(self.checkers.keys())
        
        # Run checks concurrently
        tasks = []
        for source in sources:
            if source in self.checkers:
                tasks.append(self.checkers[source].check_claim(claim))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out errors and None results
        valid_results = []
        for result in results:
            if isinstance(result, FactCheckResult):
                valid_results.append(result)
            elif isinstance(result, Exception):
                self.logger.error(f"Fact checking error: {result}")
        
        return valid_results
    
    def aggregate_results(self, results: List[FactCheckResult]) -> Dict[str, Any]:
        """Aggregate results from multiple sources"""
        if not results:
            return {
                'consensus_rating': 0.5,
                'consensus_verdict': 'Unverified',
                'confidence': 0.0,
                'sources': []
            }
        
        # Calculate weighted average rating
        total_weight = 0.0
        weighted_sum = 0.0
        
        for result in results:
            weight = self.source_weights.get(result.source.lower(), 0.8)
            weighted_sum += result.rating * weight
            total_weight += weight
        
        consensus_rating = weighted_sum / total_weight if total_weight > 0 else 0.5
        
        # Determine consensus verdict
        if consensus_rating >= 0.8:
            consensus_verdict = "True"
        elif consensus_rating >= 0.6:
            consensus_verdict = "Mostly True"
        elif consensus_rating >= 0.4:
            consensus_verdict = "Mixed"
        elif consensus_rating >= 0.2:
            consensus_verdict = "Mostly False"
        else:
            consensus_verdict = "False"
        
        # Calculate confidence based on agreement
        ratings = [r.rating for r in results]
        if len(ratings) > 1:
            variance = sum((r - consensus_rating) ** 2 for r in ratings) / len(ratings)
            confidence = max(0.0, 1.0 - (variance * 2))  # Lower confidence with higher variance
        else:
            confidence = 0.6  # Single source
        
        return {
            'consensus_rating': consensus_rating,
            'consensus_verdict': consensus_verdict,
            'confidence': confidence,
            'sources': [
                {
                    'name': r.source,
                    'verdict': r.verdict,
                    'rating': r.rating,
                    'url': r.url
                }
                for r in results
            ]
        }
    
    async def check_claim_with_consensus(self, claim: str) -> Dict[str, Any]:
        """Check claim and return consensus result"""
        results = await self.check_claim(claim)
        consensus = self.aggregate_results(results)
        
        # Add individual results
        consensus['detailed_results'] = results
        
        return consensus