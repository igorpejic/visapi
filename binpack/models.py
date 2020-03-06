from django.db import models
from django.contrib.postgres.fields import JSONField, ArrayField


PROBLEM_GENERATOR_CHOICES = [
    ('guillotine', 'guillotine'),
    ('florian', 'florian'),
]

STRATEGY_CHOICES = [
    ('max_depth', 'max_depth'),
    ('avg_depth', 'avg_depth'),
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
    n_tiles = models.IntegerField(blank=True, null=True)
    score = models.FloatField(blank=True, null=True, help_text='Tiles left')
    solution_found = models.BooleanField(default=False)

    problem_generator = models.CharField(
        max_length=100,
    choices=PROBLEM_GENERATOR_CHOICES)

    strategy = models.CharField(
        max_length=100,
        choices=STRATEGY_CHOICES, 
        default='max_depth')

    n_tiles_placed = models.IntegerField(blank=True, null=True, help_text='Number of tiles placed before a solution was found')

    their_id = models.BigIntegerField(blank=True, null=True, help_text='ID from external dataset.')
    their_tiles_placed = models.BigIntegerField(blank=True, null=True, help_text='Amount of tiles placed with heuristics algorithm')

    b_their_id = models.BigIntegerField(blank=True, null=True, help_text='ID from external dataset.')
    b_their_tiles_placed = models.BigIntegerField(blank=True, null=True, help_text='Amount of tiles placed with heuristics algorithm')

    problem_id = models.CharField(blank=True, null=True, max_length=500)
    solution_tiles_order = JSONField(blank=True, null=True)

    improved_sel = models.BooleanField(default=False)

    class Meta:
        db_table = 'affinity'
        unique_together = (('problem_generator', 'rows', 'cols', 'problem_id', 'their_id', 'improved_sel'),)
        ordering=('-created_on',)

    def __repr__(self):
        return f'{self.problem_generator} -{len(self.tiles)} ({self.rows} x {self.cols})'

    def __str__(self):
        return self.__repr__()
