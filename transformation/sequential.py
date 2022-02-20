from transformation import AbstractTransformation
from typing import List


class Sequential(AbstractTransformation):
    def __init__(self, transformations):
        self.transformations: List[AbstractTransformation] = transformations
        parameters_size_tmp = 0
        for transformation in self.transformations:
            parameters_size_tmp += transformation.parameters_size

        super().__init__(
            domain=transformations[0].domain,
            image=transformations[-1].image,
            parameters_size=parameters_size_tmp
        )

    def forward(self, inputs, parameters, *args, **kwargs):
        position = 0
        for transformation in self.transformations:
            transformation_parameters = parameters[position:position + transformation.parameters_size]
            inputs = transformation(inputs, transformation_parameters)
            position += transformation.parameters_size
        return inputs
