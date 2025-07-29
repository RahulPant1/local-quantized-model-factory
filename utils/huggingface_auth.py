#!/usr/bin/env python3
"""
HuggingFace Authentication Utility
Handles HuggingFace authentication, token management, and connection verification
"""

import os
import json
import getpass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

import requests
from huggingface_hub import HfApi, whoami
from huggingface_hub.utils import HfHubHTTPError
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.text import Text

console = Console()

@dataclass
class HFUserInfo:
    """HuggingFace user information"""
    username: str
    fullname: str
    email: Optional[str] = None
    avatar_url: Optional[str] = None
    plan: Optional[str] = None
    orgs: Optional[list] = None
    
class HuggingFaceAuth:
    """HuggingFace authentication manager"""
    
    def __init__(self, cache_dir: str = ".hf_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.token_cache_file = self.cache_dir / "hf_token.json"
        self.user_cache_file = self.cache_dir / "hf_user.json"
        
        # Initialize HF API
        self.hf_api = HfApi()
        self._current_token = None
        self._user_info = None
        
        # Load cached token if available
        self._load_cached_token()
    
    def _load_cached_token(self) -> Optional[str]:
        """Load cached HF token"""
        try:
            if self.token_cache_file.exists():
                with open(self.token_cache_file, 'r') as f:
                    data = json.load(f)
                    token = data.get('token')
                    if token:
                        self._current_token = token
                        return token
        except Exception as e:
            console.print(f"[yellow]Could not load cached token: {e}[/yellow]")
        return None
    
    def _save_token_cache(self, token: str):
        """Save token to cache"""
        try:
            cache_data = {
                'token': token,
                'saved_at': datetime.now().isoformat(),
                'source': 'lqmf'
            }
            with open(self.token_cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            console.print(f"[yellow]Could not save token cache: {e}[/yellow]")
    
    def _save_user_cache(self, user_info: HFUserInfo):
        """Save user info to cache"""
        try:
            with open(self.user_cache_file, 'w') as f:
                json.dump(asdict(user_info), f, indent=2)
        except Exception as e:
            console.print(f"[yellow]Could not save user cache: {e}[/yellow]")
    
    def get_token_from_env(self) -> Optional[str]:
        """Get HF token from environment variables"""
        token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')
        if token:
            self._current_token = token
            return token
        return None
    
    def get_token_from_huggingface_cli(self) -> Optional[str]:
        """Get token from HuggingFace CLI cache"""
        try:
            # Check HF CLI token location
            hf_cache_home = Path.home() / ".cache" / "huggingface"
            token_file = hf_cache_home / "token"
            
            if token_file.exists():
                with open(token_file, 'r') as f:
                    token = f.read().strip()
                    if token:
                        self._current_token = token
                        return token
        except Exception as e:
            console.print(f"[yellow]Could not read HF CLI token: {e}[/yellow]")
        return None
    
    def verify_token(self, token: str) -> Tuple[bool, Optional[HFUserInfo]]:
        """Verify HF token and get user info"""
        try:
            # Test token by getting user info
            user_info = whoami(token=token)
            
            if user_info:
                hf_user = HFUserInfo(
                    username=user_info.get('name', 'unknown'),
                    fullname=user_info.get('fullname', ''),
                    email=user_info.get('email'),
                    avatar_url=user_info.get('avatarUrl'),
                    plan=user_info.get('plan', {}).get('name', 'free'),
                    orgs=[org.get('name') for org in user_info.get('orgs', [])]
                )
                
                self._user_info = hf_user
                self._save_user_cache(hf_user)
                return True, hf_user
            else:
                return False, None
                
        except HfHubHTTPError as e:
            if e.response.status_code == 401:
                return False, None
            else:
                console.print(f"[red]HTTP Error: {e}[/red]")
                return False, None
        except Exception as e:
            console.print(f"[red]Error verifying token: {e}[/red]")
            return False, None
    
    def interactive_login(self) -> Optional[str]:
        """Interactive HF login process"""
        console.print(Panel.fit("🤗 HuggingFace Authentication", style="bold blue"))
        
        console.print("\n[cyan]To access private models and avoid rate limits, please authenticate with HuggingFace.[/cyan]")
        console.print("\n[dim]You can get your token from: https://huggingface.co/settings/tokens[/dim]")
        
        # Check if user wants to proceed
        if not Confirm.ask("\nWould you like to authenticate with HuggingFace?"):
            console.print("[yellow]Skipping HuggingFace authentication. Some features may be limited.[/yellow]")
            return None
        
        # Get token from user
        token = Prompt.ask(
            "\n[green]Enter your HuggingFace token[/green]",
            password=True,
            default=""
        )
        
        if not token:
            console.print("[yellow]No token provided. Skipping authentication.[/yellow]")
            return None
        
        # Verify token
        console.print("\n[blue]Verifying token...[/blue]")
        is_valid, user_info = self.verify_token(token)
        
        if is_valid and user_info:
            console.print(f"[green]✅ Authentication successful![/green]")
            console.print(f"[green]Welcome, {user_info.fullname or user_info.username}![/green]")
            
            # Save token
            self._save_token_cache(token)
            self._current_token = token
            
            # Display user info
            self.display_user_info(user_info)
            
            return token
        else:
            console.print("[red]❌ Authentication failed. Invalid token.[/red]")
            console.print("[yellow]Please check your token and try again.[/yellow]")
            return None
    
    def get_current_token(self) -> Optional[str]:
        """Get current active token"""
        # Priority order: current token -> env -> cached -> HF CLI
        if self._current_token:
            return self._current_token
        
        token = self.get_token_from_env()
        if token:
            return token
        
        token = self._load_cached_token()
        if token:
            return token
        
        token = self.get_token_from_huggingface_cli()
        if token:
            return token
        
        return None
    
    def check_authentication_status(self) -> Dict[str, Any]:
        """Check current authentication status"""
        token = self.get_current_token()
        
        if not token:
            return {
                'authenticated': False,
                'token_source': None,
                'user_info': None,
                'message': 'No HuggingFace token found'
            }
        
        # Verify current token
        is_valid, user_info = self.verify_token(token)
        
        # Determine token source
        token_source = 'unknown'
        if token == os.getenv('HF_TOKEN') or token == os.getenv('HUGGINGFACE_TOKEN'):
            token_source = 'environment'
        elif self.token_cache_file.exists():
            token_source = 'cached'
        else:
            token_source = 'hf_cli'
        
        return {
            'authenticated': is_valid,
            'token_source': token_source,
            'user_info': user_info,
            'message': f'Authenticated as {user_info.username}' if is_valid else 'Invalid token'
        }
    
    def display_user_info(self, user_info: HFUserInfo):
        """Display user information in formatted table"""
        table = Table(title="HuggingFace User Information")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Username", user_info.username)
        table.add_row("Full Name", user_info.fullname or "N/A")
        table.add_row("Email", user_info.email or "N/A")
        table.add_row("Plan", user_info.plan or "free")
        
        if user_info.orgs:
            table.add_row("Organizations", ", ".join(user_info.orgs))
        
        console.print(table)
    
    def display_auth_status(self):
        """Display current authentication status"""
        status = self.check_authentication_status()
        
        if status['authenticated']:
            console.print(Panel.fit(
                f"✅ HuggingFace Authentication Status: [green]Active[/green]\n"
                f"👤 User: {status['user_info'].username}\n"
                f"📍 Token Source: {status['token_source']}",
                style="green",
                title="🤗 HuggingFace"
            ))
        else:
            console.print(Panel.fit(
                f"❌ HuggingFace Authentication Status: [red]Not Authenticated[/red]\n"
                f"💡 Run 'hf login' to authenticate\n"
                f"🔗 Get token: https://huggingface.co/settings/tokens",
                style="red",
                title="🤗 HuggingFace"
            ))
    
    def logout(self):
        """Logout and clear cached tokens"""
        try:
            # Clear cached files
            if self.token_cache_file.exists():
                self.token_cache_file.unlink()
            if self.user_cache_file.exists():
                self.user_cache_file.unlink()
            
            # Clear in-memory data
            self._current_token = None
            self._user_info = None
            
            console.print("[green]✅ Logged out successfully![/green]")
        except Exception as e:
            console.print(f"[red]Error during logout: {e}[/red]")
    
    def test_access(self, model_name: str = "gpt2") -> bool:
        """Test access to HuggingFace Hub"""
        try:
            token = self.get_current_token()
            
            # Test API access
            model_info = self.hf_api.model_info(model_name, token=token)
            
            if model_info:
                console.print(f"[green]✅ Successfully accessed model: {model_name}[/green]")
                return True
            else:
                console.print(f"[red]❌ Could not access model: {model_name}[/red]")
                return False
                
        except Exception as e:
            console.print(f"[red]❌ Access test failed: {e}[/red]")
            return False

def main():
    """Test the HuggingFace authentication utility"""
    hf_auth = HuggingFaceAuth()
    
    # Display current status
    hf_auth.display_auth_status()
    
    # Check if authentication is needed
    status = hf_auth.check_authentication_status()
    
    if not status['authenticated']:
        # Try interactive login
        token = hf_auth.interactive_login()
        if token:
            console.print("[green]Authentication setup complete![/green]")
        else:
            console.print("[yellow]Continuing without authentication.[/yellow]")
    else:
        console.print("[green]Already authenticated![/green]")
        if status['user_info']:
            hf_auth.display_user_info(status['user_info'])
    
    # Test access
    console.print("\n[blue]Testing HuggingFace access...[/blue]")
    hf_auth.test_access()

if __name__ == "__main__":
    main()