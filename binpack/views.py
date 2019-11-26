from rest_framework import viewsets
from rest_framework import serializers
from .models import Result

class ResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = Result
        fields = '__all__'


class ResultViewSet(viewsets.ModelViewSet):
    queryset = Result.objects.all()
    serializer_class = ResultSerializer
