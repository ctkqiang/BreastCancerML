#!/usr/bin/env python
copyright = """
                  Copyright 2020 © John Melody Me

      Licensed under the Apache License, Version 2.0 (the "License");
      you may not use this file except in compliance with the License.
      You may obtain a copy of the License at

                  http://www.apache.org/licenses/LICENSE-2.0

      Unless required by applicable law or agreed to in writing, software
      distributed under the License is distributed on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
      See the License for the specific language governing permissions and
      limitations under the License.
      @Author : John Melody Me
      @Copyright: John Melody Me & Tan Sin Dee © Copyright 2020
      @INPIREDBYGF: Cindy Tan Sin Dee <3
"""

genes = open("Model/Sequence/breast.fasta", "r")
print(copyright)
# $Value to 0:
g = 0
a = 0
c = 0
t = 0

#Skip $Header():
for line in genes:
      line = line.lower()
      for char in line:
            if char == "g":
                  g += 1
            if char == "a":
                  a += 1
            if char == "c":
                  c += 1
            if char == "t":
                  t += 1
print("\n",
"Number of \"g\": ", str(g) ,"\n",
"Number of \"a\": ", str(a), "\n" ,
"Number of \"c\": ", str(c), "\n" ,
"Number of \"t\" ", str(t), "\n"
)

# 0. convert to float:
gc = (g + c + 0.) / (a + t + c + g + 0.)
print("\"gc\" Content: ", str(gc), "\n")

details = open("Model/Sequence/breast.fasta", "r")
details.read()