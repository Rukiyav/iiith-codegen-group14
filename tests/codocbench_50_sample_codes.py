"""
CoDocBench — first 50 sample codes from the documentation dataset
Source: data/codocbench/codocbench_synthetic.json
"""

# --- Sample 1: similar_elements ---
# Reference doc: Write a function to find the shared elements from the given two lists.
def similar_elements(test_tup1, test_tup2):
  res = tuple(set(test_tup1) & set(test_tup2))
  return (res)

# --- Sample 2: is_not_prime ---
# Reference doc: Write a python function to identify non-prime numbers.
import math
def is_not_prime(n):
    result = False
    for i in range(2,int(math.sqrt(n)) + 1):
        if n % i == 0:
            result = True
    return result

# --- Sample 3: heap_queue_largest ---
# Reference doc: Write a function to find the n largest integers from a given list of numbers, returned in descending order.
import heapq as hq
def heap_queue_largest(nums,n):
  largest_nums = hq.nlargest(n, nums)
  return largest_nums

# --- Sample 4: is_Power_Of_Two ---
# Reference doc: Write a python function to check whether the two numbers differ at one bit position only or not.
def is_Power_Of_Two (x): 
    return x and (not(x & (x - 1))) 
def differ_At_One_Bit_Pos(a,b): 
    return is_Power_Of_Two(a ^ b)

# --- Sample 5: find_char_long ---
# Reference doc: Write a function to find all words which are at least 4 characters long in a string.
import re
def find_char_long(text):
  return (re.findall(r"\b\w{4,}\b", text))

# --- Sample 6: square_nums ---
# Reference doc: Write a function to find squares of individual elements in a list.
def square_nums(nums):
 square_nums = list(map(lambda x: x ** 2, nums))
 return square_nums

# --- Sample 7: find_Rotations ---
# Reference doc: Write a python function to find the minimum number of rotations (greater than 0) required to get the same string.
def find_Rotations(str): 
    tmp = str + str
    n = len(str) 
    for i in range(1,n + 1): 
        substring = tmp[i: i+n] 
        if (str == substring): 
            return i 
    return n

# --- Sample 8: remove_Occ ---
# Reference doc: Write a python function to remove first and last occurrence of a given character from the string.
def remove_Occ(s,ch): 
    for i in range(len(s)): 
        if (s[i] == ch): 
            s = s[0 : i] + s[i + 1:] 
            break
    for i in range(len(s) - 1,-1,-1):  
        if (s[i] == ch): 
            s = s[0 : i] + s[i + 1:] 
            break
    return s

# --- Sample 9: sort_matrix ---
# Reference doc: Write a function to sort a given matrix in ascending order according to the sum of its rows.
def sort_matrix(M):
    result = sorted(M, key=sum)
    return result

# --- Sample 10: find_Volume ---
# Reference doc: Write a python function to find the volume of a triangular prism.
def find_Volume(l,b,h) : 
    return ((l * b * h) / 2)

# --- Sample 11: text_lowercase_underscore ---
# Reference doc: Write a function to that returns true if the input string contains sequences of lowercase letters joined with an underscore and false otherwise.
import re
def text_lowercase_underscore(text):
        patterns = '^[a-z]+_[a-z]+$'
        if re.search(patterns,  text):
                return True
        else:
                return False

# --- Sample 12: square_perimeter ---
# Reference doc: Write a function that returns the perimeter of a square given its side length as input.
def square_perimeter(a):
  perimeter=4*a
  return perimeter

# --- Sample 13: str_to_list ---
# Reference doc: Write a function to remove characters from the first string which are present in the second string.
NO_OF_CHARS = 256
def str_to_list(string): 
	temp = [] 
	for x in string: 
		temp.append(x) 
	return temp 
def lst_to_string(List): 
	return ''.join(List) 
def get_char_count_array(string): 
	count = [0] * NO_OF_CHARS 
	for i in string: 
		count[ord(i)] += 1
	return count 
def remove_dirty_chars(string, second_string): 
	count = get_char_count_array(second_string) 
	ip_ind = 0
	res_ind = 0
	temp = '' 
	str_list = str_to_list(string) 
	while ip_ind != len(str_list): 
		temp = str_list[ip_ind] 
		if count[ord(temp)] == 0: 
			str_list[res_ind] = str_list[ip_ind] 
			res_ind += 1
		ip_ind+=1
	return lst_to_string(str_list[0:res_ind])

# --- Sample 14: test_duplicate ---
# Reference doc: Write a function to find whether a given array of integers contains any duplicate element.
def test_duplicate(arraynums):
    nums_set = set(arraynums)    
    return len(arraynums) != len(nums_set)

# --- Sample 15: is_woodall ---
# Reference doc: Write a function to check if the given number is woodball or not.
def is_woodall(x): 
	if (x % 2 == 0): 
		return False
	if (x == 1): 
		return True
	x = x + 1 
	p = 0
	while (x % 2 == 0): 
		x = x/2
		p = p + 1
		if (p == x): 
			return True
	return False

# --- Sample 16: rev ---
# Reference doc: Write a python function to check if a given number is one less than twice its reverse.
def rev(num):    
    rev_num = 0
    while (num > 0):  
        rev_num = (rev_num * 10 + num % 10) 
        num = num // 10  
    return rev_num  
def check(n):    
    return (2 * rev(n) == n + 1)

# --- Sample 17: find_Max_Num ---
# Reference doc: Write a python function to find the largest number that can be formed with the given list of digits.
def find_Max_Num(arr) : 
    n = len(arr)
    arr.sort(reverse = True) 
    num = arr[0] 
    for i in range(1,n) : 
        num = num * 10 + arr[i] 
    return num

# --- Sample 18: opposite_Signs ---
# Reference doc: Write a python function to check whether the given two integers have opposite sign or not.
def opposite_Signs(x,y): 
    return ((x ^ y) < 0);

# --- Sample 19: is_octagonal ---
# Reference doc: Write a function to find the nth octagonal number.
def is_octagonal(n): 
	return 3 * n * n - 2 * n

# --- Sample 20: count_Substrings ---
# Reference doc: Write a python function to count the number of substrings with the sum of digits equal to their length.
from collections import defaultdict
def count_Substrings(s):
    n = len(s)
    count,sum = 0,0
    mp = defaultdict(lambda : 0)
    mp[0] += 1
    for i in range(n):
        sum += ord(s[i]) - ord('0')
        count += mp[sum - (i + 1)]
        mp[sum - (i + 1)] += 1
    return count

# --- Sample 21: smallest_num ---
# Reference doc: Write a python function to find smallest number in a list.
def smallest_num(xs):
  return min(xs)

# --- Sample 22: max_difference ---
# Reference doc: Write a function to find the maximum difference between available pairs in the given tuple list.
def max_difference(test_list):
  temp = [abs(b - a) for a, b in test_list]
  res = max(temp)
  return (res)

# --- Sample 23: subject_marks ---
# Reference doc: Write a function to sort a list of tuples using the second value of each tuple.
def subject_marks(subjectmarks):
#subject_marks = [('English', 88), ('Science', 90), ('Maths', 97), ('Social sciences', 82)])
 subjectmarks.sort(key = lambda x: x[1])
 return subjectmarks

# --- Sample 24: recursive_list_sum ---
# Reference doc: Write a function to flatten a list and sum all of its elements.
def recursive_list_sum(data_list):
	total = 0
	for element in data_list:
		if type(element) == type([]):
			total = total + recursive_list_sum(element)
		else:
			total = total + element
	return total

# --- Sample 25: pos_count ---
# Reference doc: Write a python function to count the number of positive numbers in a list.
def pos_count(list):
  pos_count= 0
  for num in list: 
    if num >= 0: 
      pos_count += 1
  return pos_count

# --- Sample 26: bell_number ---
# Reference doc: Write a function to find the number of ways to partition a set of Bell numbers.
def bell_number(n):   
    bell = [[0 for i in range(n+1)] for j in range(n+1)] 
    bell[0][0] = 1
    for i in range(1, n+1): 
        bell[i][0] = bell[i-1][i-1]  
        for j in range(1, i+1): 
            bell[i][j] = bell[i-1][j-1] + bell[i][j-1]   
    return bell[n][0]

# --- Sample 27: is_Monotonic ---
# Reference doc: Write a python function to check whether the given array is monotonic or not.
def is_Monotonic(A): 
    return (all(A[i] <= A[i + 1] for i in range(len(A) - 1)) or
            all(A[i] >= A[i + 1] for i in range(len(A) - 1)))

# --- Sample 28: is_sublist ---
# Reference doc: Write a function to check whether a list contains the given sublist or not.
def is_sublist(l, s):
	sub_set = False
	if s == []:
		sub_set = True
	elif s == l:
		sub_set = True
	elif len(s) > len(l):
		sub_set = False
	else:
		for i in range(len(l)):
			if l[i] == s[0]:
				n = 1
				while (n < len(s)) and (l[i+n] == s[n]):
					n += 1				
				if n == len(s):
					sub_set = True
	return sub_set

# --- Sample 29: find_equal_tuple ---
# Reference doc: Write a function to find whether all the given tuples have equal length or not.
def find_equal_tuple(Input):
  k = 0 if not Input else len(Input[0])
  flag = 1
  for tuple in Input:
    if len(tuple) != k:
      flag = 0
      break
  return flag
def get_equal(Input):
  return find_equal_tuple(Input) == 1

# --- Sample 30: comb_sort ---
# Reference doc: Write a function to sort a list of elements.
def comb_sort(nums):
    shrink_fact = 1.3
    gaps = len(nums)
    swapped = True
    i = 0
    while gaps > 1 or swapped:
        gaps = int(float(gaps) / shrink_fact)
        swapped = False
        i = 0
        while gaps + i < len(nums):
            if nums[i] > nums[i+gaps]:
                nums[i], nums[i+gaps] = nums[i+gaps], nums[i]
                swapped = True
            i += 1
    return nums

# --- Sample 31: dif_Square ---
# Reference doc: Write a python function to check whether the given number can be represented as the difference of two squares or not.
def dif_Square(n): 
    if (n % 4 != 2): 
        return True
    return False

# --- Sample 32: is_samepatterns ---
# Reference doc: Write a function to check whether it follows the sequence given in the patterns array.
def is_samepatterns(colors, patterns):    
    if len(colors) != len(patterns):
        return False    
    sdict = {}
    pset = set()
    sset = set()    
    for i in range(len(patterns)):
        pset.add(patterns[i])
        sset.add(colors[i])
        if patterns[i] not in sdict.keys():
            sdict[patterns[i]] = []

        keys = sdict[patterns[i]]
        keys.append(colors[i])
        sdict[patterns[i]] = keys

    if len(pset) != len(sset):
        return False   

    for values in sdict.values():

        for i in range(len(values) - 1):
            if values[i] != values[i+1]:
                return False

    return True

# --- Sample 33: find_tuples ---
# Reference doc: Write a function to find tuples which have all elements divisible by k from the given list of tuples.
def find_tuples(test_list, K):
  res = [sub for sub in test_list if all(ele % K == 0 for ele in sub)]
  return res

# --- Sample 34: is_Diff ---
# Reference doc: Write a python function to find whether a number is divisible by 11.
def is_Diff(n): 
    return (n % 11 == 0)

# --- Sample 35: word_len ---
# Reference doc: Write a python function to check whether the length of the word is odd or not.
def word_len(s): 
    s = s.split(' ')   
    for word in s:    
        if len(word)%2!=0: 
            return True  
        else:
          return False

# --- Sample 36: tetrahedral_number ---
# Reference doc: Write a function to find the nth tetrahedral number.
def tetrahedral_number(n): 
	return (n * (n + 1) * (n + 2)) / 6

# --- Sample 37: volume_sphere ---
# Reference doc: Write a function to find the volume of a sphere.
import math
def volume_sphere(r):
  volume=(4/3)*math.pi*r*r*r
  return volume

# --- Sample 38: get_Char ---
# Reference doc: Write a python function to find the character made by adding the ASCII value of all the characters of the given string modulo 26.
def get_Char(strr):  
    summ = 0
    for i in range(len(strr)): 
        summ += (ord(strr[i]) - ord('a') + 1)  
    if (summ % 26 == 0): 
        return ord('z') 
    else: 
        summ = summ % 26
        return chr(ord('a') + summ - 1)

# --- Sample 39: sequence ---
# Reference doc: Write a function to find the nth number in the newman conway sequence.
def sequence(n): 
	if n == 1 or n == 2: 
		return 1
	else: 
		return sequence(sequence(n-1)) + sequence(n-sequence(n-1))

# --- Sample 40: surfacearea_sphere ---
# Reference doc: Write a function to find the surface area of a sphere.
import math
def surfacearea_sphere(r):
  surfacearea=4*math.pi*r*r
  return surfacearea

# --- Sample 41: centered_hexagonal_number ---
# Reference doc: Write a function to find nth centered hexagonal number.
def centered_hexagonal_number(n):
  return 3 * n * (n - 1) + 1

# --- Sample 42: merge_dictionaries_three ---
# Reference doc: Write a function to merge three dictionaries into a single dictionary.
import collections as ct
def merge_dictionaries_three(dict1,dict2, dict3):
    merged_dict = dict(ct.ChainMap({},dict1,dict2,dict3))
    return merged_dict

# --- Sample 43: freq_count ---
# Reference doc: Write a function to get the frequency of all the elements in a list, returned as a dictionary.
import collections
def freq_count(list1):
  freq_count= collections.Counter(list1)
  return freq_count

# --- Sample 44: closest_num ---
# Reference doc: Write a function to find the closest smaller number than n.
def closest_num(N):
  return (N - 1)

# --- Sample 45: len_log ---
# Reference doc: Write a python function to find the length of the longest word.
def len_log(list1):
    max=len(list1[0])
    for i in list1:
        if len(i)>max:
            max=len(i)
    return max

# --- Sample 46: find_substring ---
# Reference doc: Write a function to check if a string is present as a substring in a given list of string values.
def find_substring(str1, sub_str):
   if any(sub_str in s for s in str1):
       return True
   return False

# --- Sample 47: is_undulating ---
# Reference doc: Write a function to check whether the given number is undulating or not.
def is_undulating(n): 
	n = str(n)
	if (len(n) <= 2): 
		return False
	for i in range(2, len(n)): 
		if (n[i - 2] != n[i]): 
			return False
	return True

# --- Sample 48: power ---
# Reference doc: Write a function to calculate the value of 'a' to the power 'b'.
def power(a,b):
	if b==0:
		return 1
	elif a==0:
		return 0
	elif b==1:
		return a
	else:
		return a*power(a,b-1)

# --- Sample 49: index_minimum ---
# Reference doc: Given a list of tuples, write a function that returns the first value of the tuple with the smallest second value.
from operator import itemgetter 
def index_minimum(test_list):
  res = min(test_list, key = itemgetter(1))[0]
  return (res)

# --- Sample 50: Find_Min_Length ---
# Reference doc: Write a python function to find the length of the smallest list in a list of lists.
def Find_Min_Length(lst):  
    minLength = min(len(x) for x in lst )
    return minLength
