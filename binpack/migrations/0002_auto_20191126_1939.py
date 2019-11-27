# Generated by Django 2.2.7 on 2019-11-26 19:39

import django.contrib.postgres.fields
import django.contrib.postgres.fields.jsonb
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('binpack', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='result',
            name='result_tree',
            field=django.contrib.postgres.fields.jsonb.JSONField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='result',
            name='tiles',
            field=django.contrib.postgres.fields.ArrayField(base_field=django.contrib.postgres.fields.ArrayField(base_field=models.IntegerField(), size=2), default=[], size=None),
            preserve_default=False,
        ),
    ]