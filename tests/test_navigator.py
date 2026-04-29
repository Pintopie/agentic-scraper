from __future__ import annotations

from agents.navigator import NavigatorAgent


def test_candidate_links_keep_product_like_urls():
    candidates = [
        {
            "href": "https://www.safcodental.com/catalog/gloves",
            "text": "Gloves",
            "class": "nav-link",
            "itemprop": "",
        },
        {
            "href": "https://www.safcodental.com/endodontics/gates-glidden-drills-peaso-reamers",
            "text": "Gates Glidden Drills & Peeso Reamers",
            "class": "product-item-link",
            "itemprop": "",
        },
        {
            "href": "https://www.safcodental.com/aquasoft-nitrile-exam-gloves.html",
            "text": "Aquasoft Nitrile Exam Gloves",
            "class": "",
            "itemprop": "url",
        },
        {
            "href": "https://www.safcodental.com/customer/account/login",
            "text": "Sign In",
            "class": "",
            "itemprop": "",
        },
    ]

    filtered = NavigatorAgent._candidate_links_for_category(candidates, "gloves")

    assert {
        candidate["href"] for candidate in filtered
    } == {
        "https://www.safcodental.com/endodontics/gates-glidden-drills-peaso-reamers",
        "https://www.safcodental.com/aquasoft-nitrile-exam-gloves.html",
    }


def test_fallback_candidates_remove_chrome_only():
    candidates = [
        {
            "href": "https://www.safcodental.com/catalog/gloves",
            "text": "Gloves",
        },
        {
            "href": "https://www.safcodental.com/endodontics/gates-glidden-drills-peaso-reamers",
            "text": "Gates Glidden Drills & Peeso Reamers",
        },
    ]

    fallback = NavigatorAgent._fallback_candidates_for_category(candidates)

    assert fallback == [
        {
            "href": "https://www.safcodental.com/catalog/gloves",
            "text": "Gloves",
        },
        {
            "href": "https://www.safcodental.com/endodontics/gates-glidden-drills-peaso-reamers",
            "text": "Gates Glidden Drills & Peeso Reamers",
        },
    ]
