from django.conf.urls import url
from . import views
from rest_framework import routers

router = routers.SimpleRouter()
router.register(r'results', views.ResultViewSet)

urlpatterns = router.urls
