#pragma once
#include <string>
#include <iostream>
#include <filesystem>
#include "R7AssetManager.h"

namespace R7 {
    namespace Core
    {
        namespace Font
        {
            inline const std::string s_defaultFontPath = "assets/Fonts/DefaultFont.ttf";

            // Cheap to copy: the bytes are shared with the asset manager's cache.
            struct FontFaceData {
                R7::AssetManager::RawBytes data;
                std::string filePath;
                bool isLoaded = false;

                size_t Size() const { return data ? data->size() : 0; }
                const char* Data() const { return data ? data->data() : nullptr; }
            };

            // Cached per path by the asset manager, so multiple fonts work.
            inline FontFaceData GetCachedFont(const std::filesystem::path& path = s_defaultFontPath) {
                FontFaceData font;
                font.filePath = path.string();

                auto result = R7::AssetManager::LoadRaw(font.filePath);
                if (result) {
                    font.data = result.value();
                    font.isLoaded = true;
                }
                else {
                    std::cout << "[R7Core::Font] Failed to load font file: " << font.filePath << "\n";
                }
                return font;
            }
        }
    }
}
