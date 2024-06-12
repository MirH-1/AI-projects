import random


digits = "0123456789"
lowerCaseLetters = "aąbcćdeęfghijklłmnńopqrsśtuwxyzż"
upperCaseLetters = "AĄBCĆDEĘFGHIJKLŁMNŃOPQRSŚTUVWXYZŻ"
specialCharacters = "!@#$%^&*()"


includeDigits = input("Include digits (Yes/No)? ").lower() == 'yes'
includeLowerCase = input("Include lower case letters (Yes/No)? ").lower() == 'yes'
includeUpperCase = input("Include upper case letters (Yes/No)? ").lower() == 'yes'
includeSpecialCharacters = input("Include special characters (Yes/No)? ").lower() == 'yes'
exclude_characters = input("Enter any character to exclude: ")

password_length = 0
while password_length < 8:
    password_length = int(input("Enter the desired length of the password (minimum 8 characters): "))
    if password_length < 8:
        print("Password length must be at least 8 characters.")

password_characters = (
    (digits if includeDigits else '') +
    (lowerCaseLetters if includeLowerCase else '') +
    (upperCaseLetters if includeUpperCase else '') +
    (specialCharacters if includeSpecialCharacters else '')
)

password_characters = ''.join(ch for ch in password_characters if ch not in exclude_characters)

while True:
    password = ''.join(random.choice(password_characters) for _ in range(password_length))
    print("Generated password:", password)

    userAccepts = input("Please let us know if you are satisfied with the password. (Yes/No): ").lower()
    if userAccepts == 'yes':
        break

print("This here would be your final password:", password)
