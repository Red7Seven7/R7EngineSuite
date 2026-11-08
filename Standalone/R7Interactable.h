#pragma once
// ==========================================================
// R7FileManager - Modern C++23 Thread-Safe File Utility Library
// Copyright (c) 2026 RedSeven (Licensed under the MIT License)
// ==========================================================

/*
* How to use
* inherit this on any class.
* call R7Interactable::Touch() for base features
* in the class function override.
* this lets you verify if said thing should be 
* touched/interacted with externally.
* set interactable to your preffered use case
* this is a comon method of creating interfaces
* in C++.
*/
namespace R7
{
	class R7Interactable
	{
	public:
		/*used for simple interactions
		buttons,etc.
		*/
		virtual void Touch() { if (!interactable)return; };
		/*used for Update Interactor operations
		interactor=whoever touched this.
		*/
		virtual void Touch(void* interactor) { if (!interactable)return; };
	protected:/*mark protected to let this be modified when inherited*/
		bool interactable = false;
	};
};

/*
for those who are new to C++
in void return functions you can use the return word to 
make your function exit early,like an emergency brake 
if conditions are met/not met
in this case its used to simply stop early if its non-interactable.
*/
/*
void pointers are just ambiguous pointers.
you know it exists but you dont know what it is.
once you convert any pointer* to void* you strip its type
but leave its location.
*/



