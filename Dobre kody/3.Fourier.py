
import tensorflow
def removeDuplicates(nums):
    i = 0
    counter = 0
    while i < len(nums) - 1:
        k = i+1
        c = 0
        while nums[i] == nums[k] and i <len(nums)-counter:
            c += 1
            counter += 1
            k += 1
        if c > 0:
            zmienna = i
            zmienna2 = i
            while c > 0:
                while zmienna < len(nums) - 1:
                    temp = nums[zmienna + 1]
                    nums[zmienna + 1] = nums[zmienna]
                    nums[zmienna] = temp
                    zmienna += 1
                c -= 1
                zmienna = zmienna2 + 1
                zmienna2 += 1
        i += 1
    print(nums)
    return len(nums) - counter

print(removeDuplicates([0,0,1,1,1,2,2,3,3,4]))