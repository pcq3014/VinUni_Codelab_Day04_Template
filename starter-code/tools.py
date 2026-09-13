import json
import os
import re
from typing import List, Dict, Any
from datetime import datetime, timedelta, timezone

RAW_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "raw-data"))

VN_TZ = timezone(timedelta(hours=7))
VALID_CATEGORIES = ("xe_dien", "du_lich")
VALID_PRIORITIES = ("low", "medium", "high")


def _load_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, data: Any) -> None:
    """Ghi ra file tạm rồi os.replace → file cũ không bị hỏng nếu quá trình ghi thất bại giữa chừng."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# Tool #1: search_product_catalog
# ---------------------------------------------------------------------------

def search_product_catalog(category: str, max_price: int = 999999999999) -> List[Dict[str, Any]]:
    """
    Tra cứu sản phẩm/dịch vụ Vingroup theo danh mục và giá tối đa.

    Args:
        category: Loại sản phẩm ('xe_dien' hoặc 'du_lich').
        max_price: Giá tối đa (VNĐ). Mặc định không giới hạn.

    Returns:
        Danh sách sản phẩm phù hợp điều kiện.
    """
    catalog_file = os.path.join(RAW_DATA_DIR, "product_catalog.json")
    if not os.path.exists(catalog_file):
        return [{"error": "Product catalog file not found."}]

    category = category.strip().lower()
    products = _load_json(catalog_file, [])
    return [
        p for p in products
        if p["category"].lower() == category and p["price_vnd"] <= max_price
    ]


# ---------------------------------------------------------------------------
# Tool #2: submit_support_ticket
# ---------------------------------------------------------------------------

def _next_sequence(tickets: List[Dict[str, Any]]) -> int:
    """Số thứ tự = số lớn nhất đang có + 1 (không dùng len() → không trùng ID khi có ticket bị xoá)."""
    sequences = [
        int(m.group(1))
        for t in tickets
        if (m := re.fullmatch(r"TK-\d{8}-(\d+)", t.get("ticket_id", "")))
    ]
    return max(sequences, default=0) + 1


def submit_support_ticket(
    customer_name: str,
    issue_description: str,
    priority: str = "medium"
) -> Dict[str, Any]:
    """
    Ghi nhận yêu cầu hỗ trợ của khách hàng vào hệ thống ticket.

    Args:
        customer_name: Tên khách hàng.
        issue_description: Mô tả vấn đề cần hỗ trợ.
        priority: Mức độ ưu tiên ('low', 'medium', 'high'). Mặc định 'medium'.

    Returns:
        Thông tin ticket vừa tạo bao gồm ticket_id, status.
    """
    customer_name = customer_name.strip()
    issue_description = issue_description.strip()
    priority = priority.strip().lower()
    if not customer_name or not issue_description:
        raise ValueError("customer_name và issue_description không được để trống.")
    if priority not in VALID_PRIORITIES:
        raise ValueError(f"priority phải là một trong {VALID_PRIORITIES}, nhận được '{priority}'.")

    tickets_file = os.path.join(RAW_DATA_DIR, "support_tickets.json")
    existing_tickets = _load_json(tickets_file, [])

    now = datetime.now(VN_TZ)
    ticket_id = f"TK-{now:%Y%m%d}-{_next_sequence(existing_tickets):03d}"
    existing_tickets.append({
        "ticket_id": ticket_id,
        "customer_name": customer_name,
        "issue_description": issue_description,
        "priority": priority,
        "status": "open",
        "created_at": now.isoformat(timespec="seconds"),
        "category": "general"
    })
    _save_json(tickets_file, existing_tickets)

    return {
        "ticket_id": ticket_id,
        "customer_name": customer_name,
        "priority": priority,
        "status": "open",
        "message": f"Ticket {ticket_id} đã được tạo thành công."
    }


# ---------------------------------------------------------------------------
# TOOL_DEFINITIONS — JSON Schemas mô tả cho LLM
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "search_product_catalog",
        "description": (
            "Tra cứu sản phẩm/dịch vụ Vingroup (xe điện VinFast, gói nghỉ dưỡng Vinpearl) "
            "theo danh mục và giá tối đa. Dùng khi khách hỏi về sản phẩm, giá hoặc lựa chọn mua."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Loại sản phẩm: 'xe_dien' (xe điện VinFast) hoặc 'du_lich' (nghỉ dưỡng Vinpearl).",
                    "enum": list(VALID_CATEGORIES)
                },
                "max_price": {
                    "type": "integer",
                    "description": "Giá tối đa tính bằng VNĐ (ví dụ 600 triệu = 600000000). Bỏ trống nếu khách không nêu giá.",
                    "minimum": 0
                }
            },
            "required": ["category"]
        }
    },
    {
        "name": "submit_support_ticket",
        "description": (
            "Tạo yêu cầu hỗ trợ khi khách báo lỗi, khiếu nại hoặc gửi phản hồi về sản phẩm/dịch vụ Vingroup. "
            "Chỉ gọi khi đã biết họ tên khách hàng."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "customer_name": {
                    "type": "string",
                    "description": "Họ tên đầy đủ của khách hàng, ví dụ 'Lê Minh Khoa'."
                },
                "issue_description": {
                    "type": "string",
                    "description": "Mô tả ngắn gọn vấn đề khách gặp phải."
                },
                "priority": {
                    "type": "string",
                    "description": "Mức độ ưu tiên: 'high' (gấp, nghiêm trọng), 'medium' (mặc định), 'low' (không gấp).",
                    "enum": list(VALID_PRIORITIES),
                    "default": "medium"
                }
            },
            "required": ["customer_name", "issue_description"]
        }
    }
]


# ---------------------------------------------------------------------------
# TOOL_MAP — Ánh xạ tên tool → hàm thực thi
# ---------------------------------------------------------------------------

TOOL_MAP = {
    "search_product_catalog": search_product_catalog,
    "submit_support_ticket": submit_support_ticket
}
