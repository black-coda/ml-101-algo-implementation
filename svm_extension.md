# The ±1 Convention in SVM - Why We Can Do This!

Excellent question! This is one of the most important conventions in SVM that often confuses learners. Let me explain why we can arbitrarily set these boundaries to ±1.

## 🎯 The Key Insight: **Scale Invariance**

The core idea: **We can always rescale our parameters without changing the decision boundary!**

### Step 1: The Basic Constraints

We start with the fundamental requirement for classification:
- For positive class (y = +1): **w·x + b ≥ 0** (on the correct side)
- For negative class (y = -1): **w·x + b ≤ 0** (on the correct side)

But this just ensures correct classification, not maximum margin.

### Step 2: Introducing a Margin

For maximum margin, we want a gap. So we require:
- Positive class: **w·x + b ≥ c** (where c > 0)
- Negative class: **w·x + b ≤ -c** (where c > 0)

### Step 3: The Rescaling Trick ✨

Here's the magic: **If (w, b) works, then (kw, kb) also works for any k > 0!**

Let's see why:

```
Original: w·x + b ≥ c for positive class
Multiply both sides by k: kw·x + kb ≥ kc

But this is just a new classifier with:
- New weights: w' = kw
- New bias: b' = kb
- New margin threshold: c' = kc
```

### Step 4: Normalizing to ±1

Since we can scale arbitrarily, we choose to **set c = 1** for convenience:

```
Original: w·x + b ≥ c
After scaling by 1/c: (w/c)·x + (b/c) ≥ 1
```

We simply **rename** these scaled parameters back to w and b:
```
w_new·x + b_new ≥ 1
```

Similarly for negative class:
```
w_new·x + b_new ≤ -1
```

## 📊 Visual Proof

```
Before Scaling:
    -c        0        +c
-----|---------|---------|-----
     |         |         |
   Class -  hyperplane  Class +
   
After Scaling (divide everything by c):
    -1        0         1
-----|---------|---------|-----
     |         |         |
   Class -  hyperplane  Class +
   
The relative positions are identical!
```

## 🔢 Numerical Example

Let's walk through an example:

### Original Classifier:
```
w = [2, 1], b = 5, c = 3

Constraints:
Positive: 2x₁ + x₂ + 5 ≥ 3
Negative: 2x₁ + x₂ + 5 ≤ -3
```

### After Scaling (divide by c=3):
```
w_new = [2/3, 1/3], b_new = 5/3, c_new = 1

Positive: (2/3)x₁ + (1/3)x₂ + 5/3 ≥ 1
Negative: (2/3)x₁ + (1/3)x₂ + 5/3 ≤ -1
```

**The decision boundary is EXACTLY the same line in space!**

## 🎨 Why ±1 is Convenient

### 1. **Simplifies Mathematics**
```
Original: yᵢ(w·xᵢ + b) ≥ c
After normalization: yᵢ(w·xᵢ + b) ≥ 1
```

### 2. **Support Vectors Become Clear**
Support vectors are exactly the points where:
```
|w·x + b| = 1
```
No messy c to carry around!

### 3. **Margin Calculation is Clean**
```
Width = 2/||w|| (not 2c/||w||)
```

## 💡 Important: The "Free Parameter" Intuition

Think of it this way:

1. **We have freedom**: For any valid separating hyperplane, we can always adjust w and b by scaling
2. **We use this freedom**: To set the margin boundaries at exactly ±1
3. **Nothing is lost**: The relative geometry stays exactly the same

It's like choosing to measure in meters instead of centimeters - the physical distance is the same, just the numbers change!

## 🔍 What If We Chose Different Values?

We could choose ±2, ±10, or any value:

```
If we choose ±k:
- Margin width = 2k/||w||
- Support vectors satisfy |w·x + b| = k
- Optimization becomes: Minimize ||w|| subject to yᵢ(w·xᵢ + b) ≥ k

But since we can rescale w by 1/k, it's equivalent to the ±1 case!
```

## 🎯 The Bottom Line

**We don't "get" ±1 from anywhere - we CHOOSE it!** 

It's a convention that:
- Makes math cleaner
- Removes redundant scaling
- Defines support vectors uniquely
- Gives us the beautiful margin formula: 2/||w||

The ±1 is just a convenient reference point, like saying "the curb is 1 unit away from the center line" - we could say 2 units, but then we'd just be measuring in half-units! 📏