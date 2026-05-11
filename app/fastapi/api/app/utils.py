from datetime import datetime
from smbclient import register_session, exists, open_file
from typing import Optional
import os
from .config import SMB_SERVER, SMB_USERNAME, SMB_PASSWORD, SMB_SHARE, SMB_PORT

_smb_session_registered = False


def init_smb_session() -> None:
    global _smb_session_registered
    if not _smb_session_registered:
        register_session(
            server=SMB_SERVER,
            username=SMB_USERNAME,
            password=SMB_PASSWORD,
            port=SMB_PORT
        )
        _smb_session_registered = True


def check_file_exists_smb(relative_path: str, share: Optional[str] = None) -> bool:
    init_smb_session()
    share = share or SMB_SHARE
    clean_path = relative_path.replace('\\', '/')
    unc_path = f"\\\\{share}\\{clean_path}"
    return exists(unc_path)


def read_file_smb(relative_path: str, share: Optional[str] = None, binary: bool = True) -> bytes:
    init_smb_session()
    share = share or SMB_SHARE
    clean_path = relative_path.replace('\\', '/')
    unc_path = f"\\\\{share}\\{clean_path}"
    mode = "rb" if binary else "r"
    with open_file(unc_path, mode=mode) as f:
        return f.read()


def format_date_for_1c(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def parse_1c_xdatetime(xdatetime_str: str) -> str:
    return xdatetime_str