# tests/conftest.py
import pytest


@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / "test_checkpoint.db")


@pytest.fixture
def sample_product_html():
    return """
    <html><body>
    <h1 class="product-title">Premium Nitrile Exam Gloves</h1>
    <span class="brand-name">Cranberry</span>
    <span class="sku-value">CRN-9004-S</span>
    <span class="price">$14.99</span>
    <div class="product-description">Powder-free nitrile gloves for dental exams.</div>
    <span class="availability">In Stock</span>
    <img class="product-image" src="https://www.safcodental.com/images/crn9004s.jpg"/>
    <span class="unit-size">100/box</span>
    </body></html>
    """


@pytest.fixture
def sample_product_json():
    return [
        {
            "productName": "Premium Nitrile Exam Gloves",
            "brand": "Cranberry",
            "itemNumber": "CRN-9004-S",
            "productUrl": "/premium-nitrile-exam-gloves.html",
            "price": "$14.99",
            "description": "Powder-free nitrile gloves.",
            "imageUrl": "https://www.safcodental.com/images/crn9004s.jpg",
            "availability": "In Stock",
        }
    ]


@pytest.fixture
def sample_listing_html():
    return """
    <html><body>
    <ul class="products-grid">
      <li class="product-item">
        <a class="product-item-link" href="/aquasoft-nitrile-exam-gloves.html">
          Aquasoft Nitrile Exam Gloves
        </a>
        <span class="price">$31.49</span>
        <span class="sku-value">AQN-100</span>
        <img src="/media/catalog/product/a/q/aquasoft.jpg"/>
      </li>
      <li class="product-item">
        <a class="product-item-link" href="/max-touch-latex-exam-gloves.html">
          MaxTouch Latex Exam Gloves
        </a>
        <span class="price">$24.00</span>
        <span class="sku-value">MTL-200</span>
      </li>
    </ul>
    <script type="application/ld+json">
    {
      "@context": "https://schema.org",
      "@type": "Product",
      "name": "JSON-LD Surgical Suture",
      "sku": "SUT-300",
      "@id": "/json-ld-surgical-suture.html",
      "image": "/media/catalog/product/s/u/suture.jpg",
      "offers": {"price": "12.50", "availability": "InStock"}
    }
    </script>
    </body></html>
    """
