from abc import ABC, abstractmethod
import math

from .w2_control_file import W2ControlFile
 
class W2Card(ABC):
 
    def __init__(self, w2_control_file: W2ControlFile, card_name: str,
        num_records: int, num_fields_list: list, value_field_width: int,
        is_left_aligned: bool):

        super().__init__()

        self.w2_control_file = w2_control_file
        self.card_name = card_name
        self.num_records = num_records
        self.num_fields_list = num_fields_list
        self.value_field_width = value_field_width
        self.is_left_aligned = is_left_aligned
        self.num_lines_list = []
        self.num_card_data_lines = 0

        for num_fields in self.num_fields_list:
            num_data_lines = int(math.ceil(num_fields/9.0))
            self.num_lines_list.append(num_data_lines)
            self.num_card_data_lines += 1
        
        self.num_card_data_lines = max(0, self.num_card_data_lines)


    
    @abstractmethod
    def do_something(self):
        pass