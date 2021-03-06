# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import os
import re # AK added this

class BookscorpusTextFormatting:
    def __init__(self, books_path, output_filename, recursive = False):
        self.books_path = books_path
        self.recursive = recursive
        self.output_filename = output_filename


    # This puts one book per line (note by AK: for Arabic books, used cp1256 as utf-8 failed)
    def merge(self):
        accents = re.compile(r'[\u064b-\u0652\u0640]') # harakaat and tatweel - AK
        arabic_punc = re.compile(r'[\u0621-\u063A\u0641-\u064A\u061b\u061f\u060c\u003A\u003D\u002E\u002F\u007C]+') # AK
        txt_rep = re.compile(r'(\[.+\]\n+)|(\(\d{1}/\d+\)\n+)|(\d+ - باب.+\n)') # AK
        with open(self.output_filename, mode='w', newline='\n') as ofile:
            for filename in glob.glob(self.books_path + '/' + '*.txt', recursive=True):
                with open(filename, mode='r', encoding='utf-8', newline='\n') as file:  # cp1256 was original Arabic books
                    for line in file:
                        line = accents.sub('',line) # ak added to remove accents
                        line = re.sub(txt_rep,'',line) # ak
                        line = ' '.join(arabic_punc.findall(line)) # ak added to remove punc
                        if line.strip() != '':
                            ofile.write(line.strip() + ' ')
                ofile.write("\n\n")
