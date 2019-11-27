from django.db import models
from django.contrib.postgres.fields import JSONField, ArrayField


PROBLEM_GENERATOR_CHOICES = [
    ('guillotine', 'guillotine'),
    ('florian', 'florian'),
]

class Result(models.Model):
    created_on = models.DateTimeField(auto_now_add=True)

    rows = models.IntegerField()
    cols = models.IntegerField()
    n_simulations = models.IntegerField()

    result_tree = JSONField(blank=True, null=True)
    tiles = ArrayField(
        ArrayField(
            models.IntegerField(),
            size=2,
        ))
    score = models.FloatField(blank=True, null=True)

    problem_generator = models.CharField(
        max_length=100,
    choices=PROBLEM_GENERATOR_CHOICES)

    class Meta:
        db_table = 'affinity'
        unique_together = (('created_on', 'problem_generator', 'rows', 'cols'),)

    def __repr__(self):
        return f'{self.created_on} -{len(self.tiles)} ({self.rows} x {self.cols})'

    def __str__(self):
        return self.__repr__()
