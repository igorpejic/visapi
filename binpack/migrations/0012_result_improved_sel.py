# Generated by Django 2.2.7 on 2020-01-18 11:04

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('binpack', '0011_auto_20200111_1038'),
    ]

    operations = [
        migrations.AddField(
            model_name='result',
            name='improved_sel',
            field=models.BooleanField(default=False),
            preserve_default=False,
        ),
    ]
