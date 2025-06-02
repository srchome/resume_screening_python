import pytest
from webapp.app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_page_loads(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b"Upload Resumes" in response.data

def test_privacy_page_loads(client):
    response = client.get('/privacy')
    assert response.status_code == 200
    assert b"Privacy Policy" in response.data

def test_404_page(client):
    response = client.get('/thispagedoesnotexist')
    assert response.status_code == 404
