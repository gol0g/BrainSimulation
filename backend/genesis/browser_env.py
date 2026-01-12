"""
Browser Environment for Autonomous Web Exploration
===================================================
Phase B: Embodied Digital Learning - Browser Component

The brain observes DOM structure, text, and visual elements,
then takes actions (click, scroll, type, navigate) to explore
and learn from the web autonomously.
"""

import asyncio
import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict
import numpy as np
import torch

try:
    from playwright.async_api import async_playwright, Browser, Page, ElementHandle
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("Warning: playwright not installed. Run: pip install playwright && playwright install chromium")


@dataclass
class DOMElement:
    """Represents an interactable DOM element"""
    tag: str
    text: str
    href: Optional[str]
    element_id: Optional[str]
    classes: List[str]
    is_visible: bool
    is_clickable: bool
    bounding_box: Optional[Dict[str, float]]
    index: int  # Position in the element list

    def to_vector(self, max_text_len: int = 50) -> np.ndarray:
        """Convert element to numerical vector for neural processing"""
        # Tag type encoding (one-hot for common tags)
        tag_types = ['a', 'button', 'input', 'div', 'span', 'p', 'h1', 'h2', 'h3', 'img', 'other']
        tag_vec = [1.0 if self.tag.lower() == t else 0.0 for t in tag_types]

        # Text features
        text_len = min(len(self.text), max_text_len) / max_text_len
        has_text = 1.0 if self.text else 0.0

        # Link features
        has_href = 1.0 if self.href else 0.0

        # Visibility/interactivity
        visible = 1.0 if self.is_visible else 0.0
        clickable = 1.0 if self.is_clickable else 0.0

        # Position features (normalized)
        if self.bounding_box:
            x = self.bounding_box.get('x', 0) / 1920  # Assume 1920 width
            y = self.bounding_box.get('y', 0) / 1080  # Assume 1080 height
            w = self.bounding_box.get('width', 0) / 1920
            h = self.bounding_box.get('height', 0) / 1080
        else:
            x, y, w, h = 0, 0, 0, 0

        return np.array(tag_vec + [text_len, has_text, has_href, visible, clickable, x, y, w, h], dtype=np.float32)


@dataclass
class BrowserObservation:
    """Complete observation of browser state"""
    url: str
    title: str
    elements: List[DOMElement]
    page_text: str
    scroll_position: float  # 0-1, how far down the page
    can_go_back: bool
    can_go_forward: bool

    def to_tensor(self, max_elements: int = 50, device: str = 'cpu') -> torch.Tensor:
        """Convert observation to tensor for neural network"""
        # URL hash (8 floats)
        url_hash = hashlib.md5(self.url.encode()).hexdigest()[:16]
        url_vec = [int(c, 16) / 15.0 for c in url_hash]

        # Page features
        page_features = [
            self.scroll_position,
            1.0 if self.can_go_back else 0.0,
            1.0 if self.can_go_forward else 0.0,
            min(len(self.elements), max_elements) / max_elements,
            min(len(self.page_text), 10000) / 10000,
        ]

        # Element vectors (padded/truncated to max_elements)
        element_vecs = []
        for i, elem in enumerate(self.elements[:max_elements]):
            element_vecs.append(elem.to_vector())

        # Pad if needed
        elem_dim = 20  # 11 tag + 9 features
        while len(element_vecs) < max_elements:
            element_vecs.append(np.zeros(elem_dim, dtype=np.float32))

        # Flatten elements
        elements_flat = np.concatenate(element_vecs)

        # Combine all features
        obs_vec = np.concatenate([
            np.array(url_vec, dtype=np.float32),
            np.array(page_features, dtype=np.float32),
            elements_flat
        ])

        return torch.tensor(obs_vec, dtype=torch.float32, device=device)


class BrowserSafetyGate:
    """
    Safety filter for browser actions
    3-level protection similar to TerminalEnv
    """

    # Allowed domains for exploration
    ALLOWED_DOMAINS = {
        'wikipedia.org',
        'wikimedia.org',
        'stackoverflow.com',
        'github.com',
        'python.org',
        'pytorch.org',
        'arxiv.org',
        'developer.mozilla.org',
        'w3schools.com',
        'docs.python.org',
        'numpy.org',
        'pandas.pydata.org',
        'scikit-learn.org',
        'localhost',
        '127.0.0.1',
    }

    # Blocked URL patterns
    BLOCKED_PATTERNS = [
        r'login',
        r'signin',
        r'signup',
        r'register',
        r'password',
        r'account',
        r'payment',
        r'checkout',
        r'cart',
        r'admin',
        r'delete',
        r'remove',
        r'\.exe$',
        r'\.msi$',
        r'\.dmg$',
        r'download(?!.*(?:pdf|html|txt))',
    ]

    # Blocked actions
    BLOCKED_ACTIONS = [
        'submit_form',
        'file_upload',
        'download_file',
        'execute_script',
    ]

    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self.blocked_attempts = []

    def is_url_safe(self, url: str) -> Tuple[bool, str]:
        """Check if URL is safe to visit"""
        url_lower = url.lower()

        # Level 1: Domain allowlist (in strict mode)
        if self.strict_mode:
            domain_ok = False
            for allowed in self.ALLOWED_DOMAINS:
                if allowed in url_lower:
                    domain_ok = True
                    break
            if not domain_ok:
                return False, f"Domain not in allowlist: {url}"

        # Level 2: Blocked patterns
        for pattern in self.BLOCKED_PATTERNS:
            if re.search(pattern, url_lower):
                return False, f"URL matches blocked pattern: {pattern}"

        # Level 3: Protocol check
        if not url_lower.startswith(('http://', 'https://', 'file://')):
            return False, f"Invalid protocol in URL: {url}"

        return True, "OK"

    def is_action_safe(self, action_type: str, target: Optional[str] = None) -> Tuple[bool, str]:
        """Check if action is safe to perform"""
        if action_type in self.BLOCKED_ACTIONS:
            return False, f"Action type blocked: {action_type}"

        return True, "OK"

    def filter(self, action_type: str, url: Optional[str] = None) -> Tuple[bool, str]:
        """Combined safety check"""
        if url:
            url_safe, url_msg = self.is_url_safe(url)
            if not url_safe:
                self.blocked_attempts.append(('url', url, url_msg))
                return False, url_msg

        action_safe, action_msg = self.is_action_safe(action_type)
        if not action_safe:
            self.blocked_attempts.append(('action', action_type, action_msg))
            return False, action_msg

        return True, "OK"


class BrowserIntrinsicReward:
    """
    FEP-based intrinsic reward for browser exploration
    Rewards curiosity, novelty, and information gain
    """

    def __init__(self):
        self.visited_urls: Set[str] = set()
        self.visited_url_hashes: Set[str] = set()
        self.seen_content_hashes: Set[str] = set()
        self.link_graph: Dict[str, Set[str]] = defaultdict(set)  # URL -> linked URLs
        self.action_history: List[str] = []
        self.knowledge_base: Dict[str, float] = {}  # topic -> familiarity

    def _hash_content(self, text: str) -> str:
        """Hash page content for novelty detection"""
        # Normalize and hash
        normalized = re.sub(r'\s+', ' ', text.lower().strip())[:5000]
        return hashlib.md5(normalized.encode()).hexdigest()

    def _extract_topics(self, text: str) -> List[str]:
        """Extract topic keywords from text"""
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        # Filter common words
        stopwords = {'this', 'that', 'with', 'from', 'have', 'been', 'were', 'they', 'their', 'what', 'when', 'where', 'which', 'would', 'could', 'should', 'about', 'there', 'these', 'those', 'being', 'other'}
        return [w for w in words if w not in stopwords][:50]

    def compute_reward(self,
                       obs: BrowserObservation,
                       action_type: str,
                       action_success: bool,
                       blocked: bool = False) -> Tuple[float, Dict[str, float]]:
        """
        Compute intrinsic reward based on:
        - Novelty: new URLs, new content
        - Information gain: new topics learned
        - Link discovery: new connections found
        - Exploration depth: going deeper into topics
        """
        reward_components = {}

        # Penalty for blocked actions
        if blocked:
            reward_components['blocked'] = -0.5
            return sum(reward_components.values()), reward_components

        # Penalty for failed actions
        if not action_success:
            reward_components['failure'] = -0.1
            return sum(reward_components.values()), reward_components

        # Base success reward
        reward_components['success'] = 0.05

        # URL novelty
        url_hash = hashlib.md5(obs.url.encode()).hexdigest()
        if url_hash not in self.visited_url_hashes:
            self.visited_url_hashes.add(url_hash)
            self.visited_urls.add(obs.url)
            reward_components['new_url'] = 0.2

        # Content novelty
        content_hash = self._hash_content(obs.page_text)
        if content_hash not in self.seen_content_hashes:
            self.seen_content_hashes.add(content_hash)
            reward_components['new_content'] = 0.15

        # Link discovery
        new_links = 0
        for elem in obs.elements:
            if elem.href and elem.href.startswith('http'):
                if elem.href not in self.link_graph[obs.url]:
                    self.link_graph[obs.url].add(elem.href)
                    new_links += 1
        if new_links > 0:
            reward_components['link_discovery'] = min(0.1 * new_links, 0.3)

        # Topic/knowledge gain
        topics = self._extract_topics(obs.page_text)
        new_topics = 0
        for topic in topics:
            if topic not in self.knowledge_base:
                self.knowledge_base[topic] = 0.1
                new_topics += 1
            else:
                # Deepen existing knowledge
                self.knowledge_base[topic] = min(1.0, self.knowledge_base[topic] + 0.01)

        if new_topics > 0:
            reward_components['new_knowledge'] = min(0.05 * new_topics, 0.25)

        # Exploration depth bonus (following links from current page)
        if action_type == 'click' and len(self.action_history) > 0:
            if self.action_history[-1] == 'click':
                reward_components['depth_bonus'] = 0.1

        # Repetition penalty
        if len(self.action_history) >= 3:
            if self.action_history[-3:] == [action_type] * 3:
                reward_components['repetition'] = -0.15

        self.action_history.append(action_type)
        if len(self.action_history) > 100:
            self.action_history = self.action_history[-100:]

        total_reward = sum(reward_components.values())
        return total_reward, reward_components

    def get_exploration_stats(self) -> Dict[str, Any]:
        """Get statistics about exploration progress"""
        return {
            'urls_visited': len(self.visited_urls),
            'unique_pages': len(self.seen_content_hashes),
            'topics_learned': len(self.knowledge_base),
            'link_graph_size': sum(len(v) for v in self.link_graph.values()),
            'top_topics': sorted(self.knowledge_base.items(), key=lambda x: -x[1])[:10],
        }


class BrowserEnv:
    """
    Gymnasium-style browser environment for autonomous exploration
    """

    # Action space
    ACTIONS = [
        'click',           # Click on element
        'scroll_down',     # Scroll page down
        'scroll_up',       # Scroll page up
        'go_back',         # Browser back
        'go_forward',      # Browser forward
        'type_search',     # Type in search box
        'navigate',        # Go to URL
        'read',            # Spend time reading (no-op with reward)
    ]

    def __init__(self,
                 headless: bool = True,
                 start_url: str = "https://en.wikipedia.org/wiki/Artificial_intelligence",
                 strict_safety: bool = True,
                 device: str = 'cpu'):

        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError("Playwright is required. Install with: pip install playwright && playwright install chromium")

        self.headless = headless
        self.start_url = start_url
        self.device = device

        self.safety_gate = BrowserSafetyGate(strict_mode=strict_safety)
        self.reward_system = BrowserIntrinsicReward()

        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.playwright = None

        self.current_obs: Optional[BrowserObservation] = None
        self.step_count = 0
        self.episode_count = 0

        # Observation/action dimensions
        self.obs_dim = 16 + 5 + 50 * 20  # URL hash + page features + elements
        self.n_actions = len(self.ACTIONS)

    async def _init_browser(self):
        """Initialize browser instance"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=self.headless)
        context = await self.browser.new_context(
            viewport={'width': 1280, 'height': 720},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        self.page = await context.new_page()

    async def _close_browser(self):
        """Close browser instance"""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    async def _get_observation(self) -> BrowserObservation:
        """Extract current browser state as observation"""
        url = self.page.url
        title = await self.page.title()

        # Get page text
        page_text = await self.page.evaluate('document.body.innerText')
        page_text = page_text[:10000] if page_text else ""

        # Get scroll position
        scroll_info = await self.page.evaluate('''
            () => ({
                scrollTop: window.scrollY,
                scrollHeight: document.body.scrollHeight,
                clientHeight: window.innerHeight
            })
        ''')
        max_scroll = max(1, scroll_info['scrollHeight'] - scroll_info['clientHeight'])
        scroll_position = scroll_info['scrollTop'] / max_scroll

        # Check navigation
        can_go_back = await self.page.evaluate('window.history.length > 1')
        can_go_forward = False  # Hard to detect reliably

        # Get interactable elements
        elements = []
        element_handles = await self.page.query_selector_all('a, button, input, [onclick], [role="button"]')

        for i, handle in enumerate(element_handles[:50]):  # Limit to 50 elements
            try:
                elem_info = await handle.evaluate('''
                    (el) => ({
                        tag: el.tagName.toLowerCase(),
                        text: (el.innerText || el.value || el.alt || '').substring(0, 100),
                        href: el.href || null,
                        id: el.id || null,
                        classes: Array.from(el.classList),
                        visible: el.offsetParent !== null,
                    })
                ''')

                box = await handle.bounding_box()

                elements.append(DOMElement(
                    tag=elem_info['tag'],
                    text=elem_info['text'].strip(),
                    href=elem_info['href'],
                    element_id=elem_info['id'],
                    classes=elem_info['classes'],
                    is_visible=elem_info['visible'],
                    is_clickable=elem_info['tag'] in ['a', 'button'] or elem_info['visible'],
                    bounding_box=box,
                    index=i
                ))
            except Exception:
                continue

        return BrowserObservation(
            url=url,
            title=title,
            elements=elements,
            page_text=page_text,
            scroll_position=scroll_position,
            can_go_back=can_go_back,
            can_go_forward=can_go_forward
        )

    async def reset(self) -> Tuple[torch.Tensor, Dict]:
        """Reset environment to start URL"""
        if not self.browser:
            await self._init_browser()

        # Navigate to start
        safe, msg = self.safety_gate.is_url_safe(self.start_url)
        if not safe:
            raise ValueError(f"Start URL blocked: {msg}")

        await self.page.goto(self.start_url, wait_until='domcontentloaded', timeout=30000)
        await asyncio.sleep(1)  # Let page settle

        self.current_obs = await self._get_observation()
        self.step_count = 0
        self.episode_count += 1

        info = {
            'url': self.current_obs.url,
            'title': self.current_obs.title,
            'n_elements': len(self.current_obs.elements),
        }

        return self.current_obs.to_tensor(device=self.device), info

    async def step(self, action: int, action_arg: Optional[Any] = None) -> Tuple[torch.Tensor, float, bool, bool, Dict]:
        """
        Execute action and return (obs, reward, terminated, truncated, info)

        action: int index into ACTIONS
        action_arg: optional argument (element index for click, text for type/navigate)
        """
        self.step_count += 1
        action_type = self.ACTIONS[action]

        # Safety check
        target_url = None
        if action_type == 'navigate' and action_arg:
            target_url = action_arg
        elif action_type == 'click' and action_arg is not None:
            # Get URL from element if it's a link
            if action_arg < len(self.current_obs.elements):
                elem = self.current_obs.elements[action_arg]
                target_url = elem.href

        safe, msg = self.safety_gate.filter(action_type, target_url)

        if not safe:
            # Blocked action
            reward, components = self.reward_system.compute_reward(
                self.current_obs, action_type, False, blocked=True
            )
            info = {
                'action': action_type,
                'blocked': True,
                'block_reason': msg,
                'reward_components': components,
            }
            return self.current_obs.to_tensor(device=self.device), reward, False, False, info

        # Execute action
        success = True
        try:
            if action_type == 'click':
                if action_arg is not None and action_arg < len(self.current_obs.elements):
                    elem = self.current_obs.elements[action_arg]
                    # Click by selector or coordinates
                    if elem.bounding_box:
                        x = elem.bounding_box['x'] + elem.bounding_box['width'] / 2
                        y = elem.bounding_box['y'] + elem.bounding_box['height'] / 2
                        await self.page.mouse.click(x, y)
                        await asyncio.sleep(0.5)
                else:
                    success = False

            elif action_type == 'scroll_down':
                await self.page.evaluate('window.scrollBy(0, 300)')

            elif action_type == 'scroll_up':
                await self.page.evaluate('window.scrollBy(0, -300)')

            elif action_type == 'go_back':
                if self.current_obs.can_go_back:
                    await self.page.go_back(wait_until='domcontentloaded', timeout=10000)
                else:
                    success = False

            elif action_type == 'go_forward':
                if self.current_obs.can_go_forward:
                    await self.page.go_forward(wait_until='domcontentloaded', timeout=10000)
                else:
                    success = False

            elif action_type == 'type_search':
                # Find search input and type
                search_input = await self.page.query_selector('input[type="search"], input[name="search"], input[name="q"]')
                if search_input and action_arg:
                    await search_input.fill(action_arg)
                    await search_input.press('Enter')
                    await asyncio.sleep(1)
                else:
                    success = False

            elif action_type == 'navigate':
                if action_arg:
                    await self.page.goto(action_arg, wait_until='domcontentloaded', timeout=30000)
                else:
                    success = False

            elif action_type == 'read':
                # Spend time "reading" - small reward for comprehension
                await asyncio.sleep(0.5)

        except Exception as e:
            success = False

        # Get new observation
        try:
            await asyncio.sleep(0.3)  # Let page update
            self.current_obs = await self._get_observation()
        except Exception:
            pass  # Keep old observation if failed

        # Compute reward
        reward, components = self.reward_system.compute_reward(
            self.current_obs, action_type, success
        )

        info = {
            'action': action_type,
            'success': success,
            'url': self.current_obs.url,
            'title': self.current_obs.title,
            'n_elements': len(self.current_obs.elements),
            'reward_components': components,
            'exploration_stats': self.reward_system.get_exploration_stats(),
        }

        # Never terminate - continuous exploration
        terminated = False
        truncated = False

        return self.current_obs.to_tensor(device=self.device), reward, terminated, truncated, info

    async def close(self):
        """Clean up resources"""
        await self._close_browser()

    def get_element_texts(self, max_elements: int = 20) -> List[str]:
        """Get text descriptions of current elements for debugging"""
        if not self.current_obs:
            return []
        return [
            f"[{i}] <{e.tag}> {e.text[:50]}{'...' if len(e.text) > 50 else ''}"
            for i, e in enumerate(self.current_obs.elements[:max_elements])
        ]


# Synchronous wrapper for easier use
class SyncBrowserEnv:
    """Synchronous wrapper around async BrowserEnv"""

    def __init__(self, **kwargs):
        self.env = BrowserEnv(**kwargs)
        self._loop = None

    def _get_loop(self):
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        return self._loop

    def reset(self):
        return self._get_loop().run_until_complete(self.env.reset())

    def step(self, action: int, action_arg: Optional[Any] = None):
        return self._get_loop().run_until_complete(self.env.step(action, action_arg))

    def close(self):
        self._get_loop().run_until_complete(self.env.close())
        if self._loop:
            self._loop.close()

    @property
    def current_obs(self):
        return self.env.current_obs

    @property
    def ACTIONS(self):
        return self.env.ACTIONS

    @property
    def obs_dim(self):
        return self.env.obs_dim

    @property
    def n_actions(self):
        return self.env.n_actions

    def get_element_texts(self, max_elements: int = 20):
        return self.env.get_element_texts(max_elements)
