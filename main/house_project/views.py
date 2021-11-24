from django.shortcuts import render
from rest_framework.views import APIView
from house_project.helpers import get_df_columns, get_result, get_test_value


class HousePrediction(APIView):
    def get(self, request):
        columns = get_df_columns()
        test_value = get_test_value()
        return render(
            request,
            "predict.html",
            {
                "data": zip(columns, test_value),
            },
        )

    def post(self, request):
        data = request.POST
        values = []
        columns = get_df_columns()
        for column in columns:
            column = column.split()[0]
            values.append(data.get(column))

        price, accuracy = get_result(values)
        print(price, accuracy)
        return render(request, "result.html", {"price": price, "accuracy": accuracy})
