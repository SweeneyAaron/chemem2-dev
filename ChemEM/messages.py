# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

class Messages:
    
    @staticmethod
    def intro(version):
        return f"""
*******************************************************************************
*                              ChemEM {version}                                   *
*                                                                             *
* Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),     *
* Hamburg, Germany.                                                           *
* Developed by Aaron Sweeney <aaron.sweeney AT cssb-hamburg.de>               *
*                                                                             *
* If you use ChemEM in your work, please cite:                                *
* ChemEM: Flexible Docking of Small Molecules in Cryo-EM Structures           *
* Aaron Sweeney, Thomas Mulvaney, Mauro Maiorca, and Maya Topf                *
* Journal of Medicinal Chemistry 2024, 67 (1), 199-212                        *
* DOI: 10.1021/acs.jmedchem.3c01134                                           *
*******************************************************************************
\n"""


    @staticmethod
    def create_centered_box(message: str, width: int = 77) -> str:
        """
        Create a centered ASCII message box (portable across Linux terminals).
    
        Example:
        +---------------------------------------------------------------------------+
        |                           Loading Config Data                            |
        +---------------------------------------------------------------------------+
        """
        # Normalize message (avoid accidental newlines breaking the box)
        message = "" if message is None else str(message).replace("\n", " ").strip()
    
        # Minimum width: "+ " + message + " +" => 4 extra chars, plus at least 2 spaces padding
        min_width = len(message) + 4
        width = max(int(width), min_width)
    
        inner_width = width - 2  # excludes the border chars
    
        # Center message inside inner width with at least one space on both sides when possible
        msg_line = message.center(inner_width)
    
        top = "*" + ("=" * inner_width) + "*"
        mid = "|" + msg_line + "|"
        bot = "*" + ("=" * inner_width) + "*"
    
        return "\n".join((top, mid, bot))


    @staticmethod 
    def chemem_warning(class_name,func_name, error):
        return f"ChemEM-Warning: Non-fatal error in {class_name}, skipping running {func_name}. \nFull Error: {error}"
    
    @staticmethod 
    def fatal_exception(class_name, error):
        return f"ChemEM-Error: Fatal error in {class_name}. \n Full Error: {error}"
    
    @staticmethod 
    def no_class_attribute_found(class_name, attr):
        return f"ChemEM-AttrError: '{class_name}' object has no attribute '{attr}'"
    
    @staticmethod 
    def overwrite(path, overwrite):
        if overwrite:
            print( f"""ChemEM-Warning: Overwriting data in {path}. 
               To stop automatic overwriting, set overwrite = 0 in the configuration file.""")
        else:
            print(f""""ChemEM-OverWriteError: {path} exists.
                  To avoid accidentally overwriting files please choose another directory or set overwrite to 1 in your configuration file""")
    
        
        