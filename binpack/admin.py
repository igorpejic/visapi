from django.contrib import admin
from django.apps import apps

from django.contrib.postgres import fields
from django_json_widget.widgets import JSONEditorWidget
from import_export.admin import ImportExportModelAdmin


class CustomModelAdmin(ImportExportModelAdmin):
    list_per_page = 100
    raw_id_fields = ()
    formfield_overrides = {
        fields.JSONField: {'widget': JSONEditorWidget},
    }

    def get_list_display(self, request):
        """
        Return a sequence containing the fields to be displayed on the
        changelist.
        """
        # Initialize raw id fields (Django does not have any function to
        # override raw_id_fields value)
        self.get_filter_horizontal(request)
        self.get_raw_id_fields(request)
        try:
            return self.model.AdminMeta.list_display
        except AttributeError:
            return self.list_display

    def get_list_filter(self, request):
        """
        Return a sequence containing the fields to be displayed as filters in
        the right sidebar of the changelist page.
        """
        try:
            return self.model.AdminMeta.list_filter
        except AttributeError:
            return self.list_filter

    def get_list_select_related(self, request):
        """
        Return a list of fields to add to the select_related() part of the
        changelist items query.
        """
        try:
            return self.model.AdminMeta.list_select_related
        except AttributeError:
            return self.list_select_related

    def get_search_fields(self, request):
        """
        Return a sequence containing the fields to be searched whenever
        somebody submits a search query.
        """
        try:
            return self.model.AdminMeta.search_fields
        except AttributeError:
            return self.search_fields

    def get_raw_id_fields(self, request):
        """
        Return a sequence containing the fields to be searched whenever
        somebody submits a search query.
        """
        try:
            self.raw_id_fields = self.model.AdminMeta.raw_id_fields
        except AttributeError:
            return self.raw_id_fields

    def get_filter_horizontal(self, request):
        """
        Return a sequence containing the fields to be searched whenever
        somebody submits a search query.
        """
        try:
            self.filter_horizontal = self.model.AdminMeta.filter_horizontal
        except AttributeError:
            pass

# Register your models here.
admin.autodiscover()
for x in apps.all_models['binpack']:
    try:
        admin.site.register(
            apps.all_models['binpack'][x],
            eval('%s%s' % (apps.all_models['binpack'][x].__name__, 'Admin'))
        )
    except:
        admin.site.register(
            apps.all_models['binpack'][x],
            CustomModelAdmin 
            # eval('%s%s' %(apps.all_models['clients'][x].__name__, 'Admin'))
        )
